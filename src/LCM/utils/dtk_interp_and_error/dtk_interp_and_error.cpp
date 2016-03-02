//---------------------------------------------------------------------------//
/*!
 * \file   interpolation_error.cpp
 * \author Irina Tezaur
 * \brief  Discrete l2 error calculation using DTK
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <ctime>
#include <cstdlib>

#include "DTK_STKMeshHelpers.hpp"
#include "DTK_STKMeshManager.hpp"
#include "DTK_MapOperatorFactory.hpp"

#include <Teuchos_GlobalMPISession.hpp>
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_XMLParameterListCoreHelpers.hpp"
#include "Teuchos_ParameterList.hpp"
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_OpaqueWrapper.hpp>
#include <Teuchos_TypeTraits.hpp>
#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_StandardCatchMacros.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <Tpetra_MultiVector.hpp>

#include <Intrepid_FieldContainer.hpp>

#include <stk_util/parallel/Parallel.hpp>

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_topology/topology.hpp>

#include <stk_io/IossBridge.hpp>
#include <stk_io/StkMeshIoBroker.hpp>

#include <Ionit_Initializer.h>
#include <Ioss_SubSystem.h>


//#define DEBUG_OUTPUT

int main(int argc, char* argv[])
{
    // INITIALIZATION
    // --------------

    // Setup communication.
    Teuchos::GlobalMPISession mpiSession(&argc,&argv);

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();

    // Read in command line options.
    std::string xml_input_filename;
    Teuchos::CommandLineProcessor clp(false);
    clp.setOption( "xml-in-file",
		   &xml_input_filename,
		   "The XML file to read into a parameter list" );
    clp.parse(argc,argv);
    
    Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::fancyOStream(Teuchos::VerboseObjectBase::getDefaultOStream());

    // Build the parameter list from the xml input.
    Teuchos::RCP<Teuchos::ParameterList> plist =
	Teuchos::rcp( new Teuchos::ParameterList() );
    Teuchos::updateParametersFromXmlFile(
	xml_input_filename, Teuchos::inoutArg(*plist) );

    // Read command-line options
    std::string source_mesh_input_file = 
	plist->get<std::string>("Source Mesh Input File");
    int src_snap_no = 
	plist->get<int>("Source Mesh Snapshot Number", 1); //this value is 1-based 
    std::string target_mesh_input_file = 
	plist->get<std::string>("Target Mesh Input File");
    std::string target_mesh_output_file = 
	plist->get<std::string>("Target Mesh Output File");
    int tgt_snap_no = 
	plist->get<int>("Target Mesh Snapshot Number", 1); //this value is 1-based
    std::string field_name = 
	plist->get<std::string>("Field Name", "solution");
    std::string field_type = 
	plist->get<std::string>("Field Type", "Node Vector");
    int field_type_num; 
    if (field_type == "Node Vector") 
      field_type_num = 0; 
    else if (field_type == "Node Scalar") 
      field_type_num = 1;
    else if (field_type == "Node Tensor") 
      field_type_num = 2;
    else 
       TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            std::endl << "Error in interpolation_error.cpp: invalid field_type = " << field_type  
            <<  "!  Valid field_types are 'Node Vector' and 'Node Scalar'." << std::endl);
    std::string src_field_name = field_name+"_src"; 
    std::string tgt_interp_field_name = field_name+"Ref";
    std::string rel_err_field_name = field_name+"RelErr";
    std::string abs_err_field_name = field_name+"AbsErr";
    
 

    // Get the raw mpi communicator (basic typedef in STK).
    Teuchos::RCP<const Teuchos::MpiComm<int> > mpi_comm = 
	Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int> >( comm );
    Teuchos::RCP<const Teuchos::OpaqueWrapper<MPI_Comm> > opaque_comm = 
	mpi_comm->getRawMpiComm();
    stk::ParallelMachine parallel_machine = (*opaque_comm)();
      

    // SOURCE MESH READ
    // ----------------
    stk::io::StkMeshIoBroker src_broker( parallel_machine );
    std::size_t src_input_index = src_broker.add_mesh_database(
  	  source_mesh_input_file, "exodus", stk::io::READ_MESH );
    src_broker.set_active_mesh(src_input_index);
    src_broker.create_input_mesh();
    //number of intervals to divide each input time step into 
    int interpolation_intervals = 1;
    stk::io::MeshField::TimeMatchOption tmo = stk::io::MeshField::CLOSEST;
    if (interpolation_intervals > 1) {
      tmo = stk::io::MeshField::LINEAR_INTERPOLATION;
    }
    src_broker.add_all_mesh_fields_as_input_fields(tmo);
    src_broker.populate_bulk_data();
    Teuchos::RCP<stk::mesh::BulkData> src_bulk_data = Teuchos::rcpFromRef( src_broker.bulk_data() );
    
    stk::util::ParameterList parameters;
    Teuchos::RCP<Ioss::Region> io_region = src_broker.get_input_io_region();
    STKIORequire(!Teuchos::is_null(io_region));

    //Get number of time steps in source mesh 
    int timestep_count = io_region->get_property("state_count").get_int();
#ifdef DEBUG_OUTPUT
    *out << "timestep_count in source mesh: " << timestep_count << std::endl;
#endif
    int step =  src_snap_no; 
    if (step > timestep_count) 
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            std::endl << "Invalid value of Source Mesh Snapshot Number = " << src_snap_no <<
                         " > total number of snapshots in "  << source_mesh_input_file 
                      << " = " << timestep_count << "." << std::endl;);
    if (step <= 0) 
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            std::endl << "Invalid value of Source Mesh Snapshot Number = " << src_snap_no <<
                         "; value must be > 0." << std::endl;);  
    if (timestep_count > 0 ) { 
      double time = io_region->get_state_time(step);
      if (step == timestep_count)
        interpolation_intervals = 1;

      int step_end = step < timestep_count ? step+1 : step;
      double tend =  io_region->get_state_time(step_end);
      double tbeg = time;
      double delta = (tend - tbeg) / static_cast<double>(interpolation_intervals);

      for (int interval = 0; interval < interpolation_intervals; interval++) {
        time = tbeg + delta * static_cast<double>(interval);
        src_broker.read_defined_input_fields(time);
      }
    }
#ifdef DEBUG_OUTPUT
    //Print which fields are found 
    const Ioss::ElementBlockContainer& elem_blocks = io_region->get_element_blocks();
    Ioss::NameList exo_fld_names;
    elem_blocks[0]->field_describe(&exo_fld_names);
    for (std::size_t i = 0; i < exo_fld_names.size(); i++){
      *out << "Found field \"" << exo_fld_names[i] << "\" in source exodus file" << std::endl; }
#endif

    // DEFINE PARTS/SELECTOR
    // ----------------

    stk::mesh::Selector src_stk_selector = stk::mesh::Selector(src_broker.meta_data().universal_part()); 
    stk::mesh::BucketVector src_part_buckets = src_stk_selector.get_buckets( stk::topology::NODE_RANK );
    std::vector<stk::mesh::Entity> src_part_nodes;
    stk::mesh::get_selected_entities(src_stk_selector, src_part_buckets, src_part_nodes );
    Intrepid::FieldContainer<double> src_node_coords =
	  DataTransferKit::STKMeshHelpers::getEntityNodeCoordinates(
	  Teuchos::Array<stk::mesh::Entity>(src_part_nodes), *src_bulk_data );

    // TARGET MESH READ
    // ----------------

    // Load the target mesh.
    stk::io::StkMeshIoBroker tgt_broker( parallel_machine );
    std::size_t tgt_input_index = tgt_broker.add_mesh_database(target_mesh_input_file, "exodus", stk::io::READ_MESH );
    tgt_broker.set_active_mesh( tgt_input_index );
    tgt_broker.create_input_mesh();
    tgt_broker.add_all_mesh_fields_as_input_fields(tmo);
      
   
    //The following switch statement can be avoided by templating on the FieldType. 
    //But we don't do that here...
    switch(field_type_num) { 

      case 0: //VectorFieldType 
      { 
        typedef stk::mesh::Field<double, stk::mesh::Cartesian> VectorFieldType ;
        //Get source_field from source mesh  
        VectorFieldType* source_field = src_broker.meta_data().get_field<VectorFieldType>(stk::topology::NODE_RANK, field_name); 
        if (source_field != 0) 
          *out << "Vector Field with name " << field_name << " found in source mesh file!" << std::endl; 
        else   
          TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            std::endl << "Vector Field with name " << field_name << " NOT found in source mesh file!" << std::endl); 

        int neq = source_field->max_size(stk::topology::NODE_RANK); 
#ifdef DEBUG_OUTPUT
        *out << "Source field has " << neq << " dofs/node." << std::endl; 
#endif  

        //Put fields on target mesh 
        // Add a nodal field to the interpolated target part.
        VectorFieldType& target_interp_field = tgt_broker.meta_data().declare_field<VectorFieldType>( 
    	        stk::topology::NODE_RANK, tgt_interp_field_name );
        stk::mesh::put_field( target_interp_field, tgt_broker.meta_data().universal_part());
    
        // Add a absolute error nodal field to the target part.
        VectorFieldType& target_abs_error_field = tgt_broker.meta_data().declare_field<VectorFieldType>( 
    	    stk::topology::NODE_RANK, abs_err_field_name );
        stk::mesh::put_field( target_abs_error_field, tgt_broker.meta_data().universal_part());

        // Add a relative error nodal field to the target part.
        VectorFieldType& target_rel_error_field = tgt_broker.meta_data().declare_field<VectorFieldType>( 
    	    stk::topology::NODE_RANK, rel_err_field_name );
        stk::mesh::put_field( target_rel_error_field, tgt_broker.meta_data().universal_part());

        // Create the target bulk data.
        tgt_broker.populate_bulk_data();
        Teuchos::RCP<stk::mesh::BulkData> tgt_bulk_data = Teuchos::rcpFromRef( tgt_broker.bulk_data() );
    
        // Add a nodal field to the interpolated target part.
        //Populate target_field 
        VectorFieldType* target_field = tgt_broker.meta_data().get_field<VectorFieldType>(stk::topology::NODE_RANK, field_name); 
        if (target_field != 0) 
          *out << "Vector Field with name " << field_name << " found in target mesh file!" << std::endl; 
        else   
          TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            std::endl << "Vector Field with name " << field_name << " NOT found in target mesh file!" << std::endl); 
    
        io_region = tgt_broker.get_input_io_region();
        STKIORequire(!Teuchos::is_null(io_region));

        //Get number of time steps in source mesh 
        timestep_count = io_region->get_property("state_count").get_int();
#ifdef DEBUG_OUTPUT
        *out << "timestep_count in target mesh: " << timestep_count << std::endl;
#endif
        step =  tgt_snap_no; 
        if (step > timestep_count) 
          TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            std::endl << "Invalid value of Target Mesh Snapshot Number = " << tgt_snap_no <<
                         " > total number of snapshots in "  << source_mesh_input_file 
                      << " = " << timestep_count << "." << std::endl;);
        if (step <= 0) 
          TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            std::endl << "Invalid value of Target Mesh Snapshot Number = " << tgt_snap_no <<
                         "; value must be > 0." << std::endl;);  
        if (timestep_count > 0 ) { 
          double time = io_region->get_state_time(step);
          if (step == timestep_count)
            interpolation_intervals = 1;

          int step_end = step < timestep_count ? step+1 : step;
          double tend =  io_region->get_state_time(step_end);
          double tbeg = time;
          double delta = (tend - tbeg) / static_cast<double>(interpolation_intervals);

          for (int interval = 0; interval < interpolation_intervals; interval++) {
            time = tbeg + delta * static_cast<double>(interval);
            tgt_broker.read_defined_input_fields(time);
          }
        }

 
        // SOLUTION TRANSFER SETUP
        // -----------------------
    
        // Create a manager for the source part elements.
        DataTransferKit::STKMeshManager src_manager( src_bulk_data, src_stk_selector );

        // Create a manager for the target part nodes.
        stk::mesh::Selector tgt_stk_selector = stk::mesh::Selector(tgt_broker.meta_data().universal_part()); 
        DataTransferKit::STKMeshManager tgt_manager( tgt_bulk_data, tgt_stk_selector );

        // Create a solution vector for the source.
        Teuchos::RCP<Tpetra::MultiVector<double, int, DataTransferKit::SupportId>> src_vector =
          src_manager.createFieldMultiVector<VectorFieldType>(Teuchos::ptr(source_field), neq);

        // Create a solution vector for the target.
        Teuchos::RCP<Tpetra::MultiVector<double,int,DataTransferKit::SupportId> > tgt_vector =
	  tgt_manager.createFieldMultiVector<VectorFieldType>(
	  Teuchos::ptr(&target_interp_field), neq);

#ifdef DEBUG_OUTPUT
        // Print out source mesh info.
        Teuchos::RCP<Teuchos::Describable> src_describe = src_manager.functionSpace()->entitySet();
        *out << "Source Mesh: " << std::endl;
        src_describe->describe(*out, Teuchos::VERB_HIGH );
        *out << std::endl;

        // Print out target mesh info.
        Teuchos::RCP<Teuchos::Describable> tgt_describe = tgt_manager.functionSpace()->entitySet();
        *out << "Target Mesh: " << std::endl;
        tgt_describe->describe(*out, Teuchos::VERB_HIGH );
        *out << std::endl;
#endif

        // SOLUTION TRANSFER
        // -----------------

        // Create a map operator. The operator settings are in the
        // "DataTransferKit" parameter list.
        Teuchos::ParameterList& dtk_list = plist->sublist("DataTransferKit");    
        DataTransferKit::MapOperatorFactory op_factory;
        Teuchos::RCP<DataTransferKit::MapOperator> map_op = op_factory.create( src_vector->getMap(),
  	  		     tgt_vector->getMap(),
			     dtk_list );

        // Setup the map operator. This creates the underlying linear operators.
        map_op->setup( src_manager.functionSpace(), tgt_manager.functionSpace() );
    
        // Apply the map operator. This interpolates the data from one STK field
        // to the other.
        map_op->apply( *src_vector, *tgt_vector );
     
#ifdef DEBUG_OUTPUT
        *out << "src_vector: \n ";
        src_vector->describe(*out, Teuchos::VERB_EXTREME);
        *out << "tgt_vector: \n ";
        tgt_vector->describe(*out, Teuchos::VERB_EXTREME);
#endif
      
        // COMPUTE THE SOLUTION ERROR
        // --------------------------

        double* tgt_field_data;
        double* gold_value; //reference solution (i.e., target_interp_field) 
        double* rel_err_field_data;
        double* abs_err_field_data;
        std::vector< stk::mesh::Entity > tgt_ownednodes ;
        stk::mesh::Selector select_owned_in_part = stk::mesh::Selector( tgt_broker.meta_data().universal_part() ) &
                                               stk::mesh::Selector( tgt_broker.meta_data().locally_owned_part() );
        stk::mesh::get_selected_entities( select_owned_in_part ,
            tgt_broker.bulk_data().buckets( stk::topology::NODE_RANK ) ,
            tgt_ownednodes );
        int tgt_num_owned_nodes = tgt_ownednodes.size(); //number owned nodes
        stk::mesh::BucketVector tgt_part_buckets = tgt_stk_selector.get_buckets( stk::topology::NODE_RANK );
        std::vector<stk::mesh::Entity> tgt_part_nodes;
        stk::mesh::get_selected_entities(tgt_stk_selector, tgt_part_buckets, tgt_part_nodes );
        Intrepid::FieldContainer<double> tgt_node_coords = DataTransferKit::STKMeshHelpers::getEntityNodeCoordinates(
              Teuchos::Array<stk::mesh::Entity>(tgt_part_nodes), *tgt_bulk_data );
        int num_tgt_part_nodes = tgt_part_nodes.size(); //number nodes (owned + overlap) 
#ifdef DEBUG_OUTPUT
        std::cout << "proc #: " << comm->getRank() << ", tgt_num_owned_nodes = " << tgt_num_owned_nodes << std::endl; 
#endif

        double error_l2_norm_sq;
        double field_l2_norm_sq;
        for (int component = 0; component < neq; component++) {
          error_l2_norm_sq = 0.0; 
          field_l2_norm_sq = 0.0;
          
          for ( int n = 0; n < tgt_num_owned_nodes; ++n )
          {
            gold_value = stk::mesh::field_data( target_interp_field, tgt_ownednodes[n] );
            tgt_field_data = stk::mesh::field_data( *target_field, tgt_ownednodes[n] );
            rel_err_field_data = stk::mesh::field_data( target_rel_error_field, tgt_ownednodes[n] );
            rel_err_field_data[component] = std::abs(tgt_field_data[component] - gold_value[component]); 
            error_l2_norm_sq += rel_err_field_data[component] * rel_err_field_data[component];
            field_l2_norm_sq += tgt_field_data[component] * tgt_field_data[component];
          }
          *out << "Dof = " << component << std::endl;
          double error_l2_norm_global, field_l2_norm_global; 
          Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &error_l2_norm_sq, &error_l2_norm_global); 
          Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &field_l2_norm_sq, &field_l2_norm_global); 
          error_l2_norm_global = std::sqrt(error_l2_norm_global); 
          field_l2_norm_global = std::sqrt(field_l2_norm_global); 
          *out << "|e|_2 (abs error): " << error_l2_norm_global << std::endl; 
          *out << "|f|_2 (norm ref soln): " << field_l2_norm_global << std::endl; 
          *out << "|e|_2 / |f|_2 (rel error): " << error_l2_norm_global / field_l2_norm_global << std::endl;

          for ( int n = 0; n < num_tgt_part_nodes; ++n )
          {
            gold_value = stk::mesh::field_data( target_interp_field, tgt_part_nodes[n] );
            tgt_field_data = stk::mesh::field_data( *target_field, tgt_part_nodes[n] );
            rel_err_field_data = stk::mesh::field_data( target_rel_error_field, tgt_part_nodes[n] );
            abs_err_field_data = stk::mesh::field_data( target_abs_error_field, tgt_part_nodes[n] );
            rel_err_field_data[component] = std::abs(tgt_field_data[component] - gold_value[component]);
            abs_err_field_data[component] = std::abs(tgt_field_data[component] - gold_value[component]);
            if (gold_value[component] != 0)
              rel_err_field_data[component] /= std::abs(gold_value[component]);
#ifdef DEBUG_OUTPUT
              *out << "tgt_field_data, gold_value, abs_err, rel_err: "
                   << tgt_field_data[component] << ", " << gold_value[component] << ", " << abs_err_field_data[component]
                   << ", " << rel_err_field_data[component] << std::endl;
#endif
          }
        }

        // TARGET MESH WRITE
        // -----------------
        std::size_t tgt_output_index = tgt_broker.create_output_mesh(
            target_mesh_output_file, stk::io::WRITE_RESULTS );
        tgt_broker.add_field( tgt_output_index, target_interp_field );
        tgt_broker.add_field( tgt_output_index, target_rel_error_field );
        tgt_broker.add_field( tgt_output_index, target_abs_error_field );
        tgt_broker.add_field( tgt_output_index, *target_field );
        tgt_broker.begin_output_step( tgt_output_index, 0.0 );
        tgt_broker.write_defined_output_fields( tgt_output_index );
        tgt_broker.end_output_step( tgt_output_index );
        break; 
      }

      case 1: //ScalarFieldType 
      { 
        typedef stk::mesh::Field<double> ScalarFieldType ;
        //Get source_field from source mesh  
        ScalarFieldType* source_field = src_broker.meta_data().get_field<ScalarFieldType>(stk::topology::NODE_RANK, field_name); 
        if (source_field != 0) 
          *out << "Scalar Field with name " << field_name << " found in source mesh file!" << std::endl; 
        else   
          TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            std::endl << "Scalar Field with name " << field_name << " NOT found in source mesh file!" << std::endl); 

        int neq = source_field->max_size(stk::topology::NODE_RANK); 
#ifdef DEBUG_OUTPUT
        *out << "Source field has " << neq << " dofs/node." << std::endl; 
#endif  

        //Put fields on target mesh 
        // Add a nodal field to the interpolated target part.
        ScalarFieldType& target_interp_field = tgt_broker.meta_data().declare_field<ScalarFieldType>( 
    	        stk::topology::NODE_RANK, tgt_interp_field_name );
        stk::mesh::put_field( target_interp_field, tgt_broker.meta_data().universal_part());
    
        // Add a absolute error nodal field to the target part.
        ScalarFieldType& target_abs_error_field = tgt_broker.meta_data().declare_field<ScalarFieldType>( 
    	    stk::topology::NODE_RANK, abs_err_field_name );
        stk::mesh::put_field( target_abs_error_field, tgt_broker.meta_data().universal_part());

        // Add a relative error nodal field to the target part.
        ScalarFieldType& target_rel_error_field = tgt_broker.meta_data().declare_field<ScalarFieldType>( 
    	    stk::topology::NODE_RANK, rel_err_field_name );
        stk::mesh::put_field( target_rel_error_field, tgt_broker.meta_data().universal_part());

        // Create the target bulk data.
        tgt_broker.populate_bulk_data();
        Teuchos::RCP<stk::mesh::BulkData> tgt_bulk_data = Teuchos::rcpFromRef( tgt_broker.bulk_data() );
    
        // Add a nodal field to the interpolated target part.
        //Populate target_field 
        ScalarFieldType* target_field = tgt_broker.meta_data().get_field<ScalarFieldType>(stk::topology::NODE_RANK, field_name); 
        if (target_field != 0) 
          *out << "Scalar Field with name " << field_name << " found in target mesh file!" << std::endl; 
        else   
          TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            std::endl << "Scalar Field with name " << field_name << " NOT found in target mesh file!" << std::endl); 
    
        io_region = tgt_broker.get_input_io_region();
        STKIORequire(!Teuchos::is_null(io_region));

        //Get number of time steps in source mesh 
        timestep_count = io_region->get_property("state_count").get_int();
#ifdef DEBUG_OUTPUT
        *out << "timestep_count in target mesh: " << timestep_count << std::endl;
#endif
        step =  tgt_snap_no; 
        if (step > timestep_count) 
          TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            std::endl << "Invalid value of Target Mesh Snapshot Number = " << tgt_snap_no <<
                         " > total number of snapshots in "  << source_mesh_input_file 
                      << " = " << timestep_count << "." << std::endl;);
        if (step <= 0) 
          TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            std::endl << "Invalid value of Target Mesh Snapshot Number = " << tgt_snap_no <<
                         "; value must be > 0." << std::endl;);  
        if (timestep_count > 0 ) { 
          double time = io_region->get_state_time(step);
          if (step == timestep_count)
            interpolation_intervals = 1;

          int step_end = step < timestep_count ? step+1 : step;
          double tend =  io_region->get_state_time(step_end);
          double tbeg = time;
          double delta = (tend - tbeg) / static_cast<double>(interpolation_intervals);

          for (int interval = 0; interval < interpolation_intervals; interval++) {
            time = tbeg + delta * static_cast<double>(interval);
            tgt_broker.read_defined_input_fields(time);
          }
        }

 
        // SOLUTION TRANSFER SETUP
        // -----------------------
    
        // Create a manager for the source part elements.
        DataTransferKit::STKMeshManager src_manager( src_bulk_data, src_stk_selector );

        // Create a manager for the target part nodes.
        stk::mesh::Selector tgt_stk_selector = stk::mesh::Selector(tgt_broker.meta_data().universal_part()); 
        DataTransferKit::STKMeshManager tgt_manager( tgt_bulk_data, tgt_stk_selector );

        // Create a solution vector for the source.
        Teuchos::RCP<Tpetra::MultiVector<double, int, DataTransferKit::SupportId>> src_vector =
          src_manager.createFieldMultiVector<ScalarFieldType>(Teuchos::ptr(source_field), neq);

        // Create a solution vector for the target.
        Teuchos::RCP<Tpetra::MultiVector<double,int,DataTransferKit::SupportId> > tgt_vector =
	  tgt_manager.createFieldMultiVector<ScalarFieldType>(
	  Teuchos::ptr(&target_interp_field), neq);

#ifdef DEBUG_OUTPUT
        // Print out source mesh info.
        Teuchos::RCP<Teuchos::Describable> src_describe = src_manager.functionSpace()->entitySet();
        *out << "Source Mesh: " << std::endl;
        src_describe->describe(*out, Teuchos::VERB_HIGH );
        *out << std::endl;

        // Print out target mesh info.
        Teuchos::RCP<Teuchos::Describable> tgt_describe = tgt_manager.functionSpace()->entitySet();
        *out << "Target Mesh: " << std::endl;
        tgt_describe->describe(*out, Teuchos::VERB_HIGH );
        *out << std::endl;
#endif

        // SOLUTION TRANSFER
        // -----------------

        // Create a map operator. The operator settings are in the
        // "DataTransferKit" parameter list.
        Teuchos::ParameterList& dtk_list = plist->sublist("DataTransferKit");    
        DataTransferKit::MapOperatorFactory op_factory;
        Teuchos::RCP<DataTransferKit::MapOperator> map_op = op_factory.create( src_vector->getMap(),
  	  		     tgt_vector->getMap(),
			     dtk_list );

        // Setup the map operator. This creates the underlying linear operators.
        map_op->setup( src_manager.functionSpace(), tgt_manager.functionSpace() );
    
        // Apply the map operator. This interpolates the data from one STK field
        // to the other.
        map_op->apply( *src_vector, *tgt_vector );
     
#ifdef DEBUG_OUTPUT
        *out << "src_vector: \n ";
        src_vector->describe(*out, Teuchos::VERB_EXTREME);
        *out << "tgt_vector: \n ";
        tgt_vector->describe(*out, Teuchos::VERB_EXTREME);
#endif
      
        // COMPUTE THE SOLUTION ERROR
        // --------------------------

        double* tgt_field_data;
        double* gold_value; //reference solution (i.e., target_interp_field) 
        double* rel_err_field_data;
        double* abs_err_field_data;
        std::vector< stk::mesh::Entity > tgt_ownednodes ;
        stk::mesh::Selector select_owned_in_part = stk::mesh::Selector( tgt_broker.meta_data().universal_part() ) &
                                               stk::mesh::Selector( tgt_broker.meta_data().locally_owned_part() );
        stk::mesh::get_selected_entities( select_owned_in_part ,
            tgt_broker.bulk_data().buckets( stk::topology::NODE_RANK ) ,
            tgt_ownednodes );
        int tgt_num_owned_nodes = tgt_ownednodes.size(); //number owned nodes
        stk::mesh::BucketVector tgt_part_buckets = tgt_stk_selector.get_buckets( stk::topology::NODE_RANK );
        std::vector<stk::mesh::Entity> tgt_part_nodes;
        stk::mesh::get_selected_entities(tgt_stk_selector, tgt_part_buckets, tgt_part_nodes );
        Intrepid::FieldContainer<double> tgt_node_coords = DataTransferKit::STKMeshHelpers::getEntityNodeCoordinates(
              Teuchos::Array<stk::mesh::Entity>(tgt_part_nodes), *tgt_bulk_data );
        int num_tgt_part_nodes = tgt_part_nodes.size(); //number nodes (owned + overlap) 
#ifdef DEBUG_OUTPUT
        std::cout << "proc #: " << comm->getRank() << ", tgt_num_owned_nodes = " << tgt_num_owned_nodes << std::endl; 
#endif

        double error_l2_norm_sq;
        double field_l2_norm_sq;
        for (int component = 0; component < neq; component++) {
          error_l2_norm_sq = 0.0; 
          field_l2_norm_sq = 0.0;
          
          for ( int n = 0; n < tgt_num_owned_nodes; ++n )
          {
            gold_value = stk::mesh::field_data( target_interp_field, tgt_ownednodes[n] );
            tgt_field_data = stk::mesh::field_data( *target_field, tgt_ownednodes[n] );
            rel_err_field_data = stk::mesh::field_data( target_rel_error_field, tgt_ownednodes[n] );
            rel_err_field_data[component] = std::abs(tgt_field_data[component] - gold_value[component]); 
            error_l2_norm_sq += rel_err_field_data[component] * rel_err_field_data[component];
            field_l2_norm_sq += tgt_field_data[component] * tgt_field_data[component];
          }
          *out << "Dof = " << component << std::endl;
          double error_l2_norm_global, field_l2_norm_global; 
          Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &error_l2_norm_sq, &error_l2_norm_global); 
          Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &field_l2_norm_sq, &field_l2_norm_global); 
          error_l2_norm_global = std::sqrt(error_l2_norm_global); 
          field_l2_norm_global = std::sqrt(field_l2_norm_global); 
          *out << "|e|_2 (abs error): " << error_l2_norm_global << std::endl; 
          *out << "|f|_2 (norm ref soln): " << field_l2_norm_global << std::endl; 
          *out << "|e|_2 / |f|_2 (rel error): " << error_l2_norm_global / field_l2_norm_global << std::endl;

          for ( int n = 0; n < num_tgt_part_nodes; ++n )
          {
            gold_value = stk::mesh::field_data( target_interp_field, tgt_part_nodes[n] );
            tgt_field_data = stk::mesh::field_data( *target_field, tgt_part_nodes[n] );
            rel_err_field_data = stk::mesh::field_data( target_rel_error_field, tgt_part_nodes[n] );
            abs_err_field_data = stk::mesh::field_data( target_abs_error_field, tgt_part_nodes[n] );
            rel_err_field_data[component] = std::abs(tgt_field_data[component] - gold_value[component]);
            abs_err_field_data[component] = std::abs(tgt_field_data[component] - gold_value[component]);
            if (gold_value[component] != 0)
              rel_err_field_data[component] /= std::abs(gold_value[component]);
#ifdef DEBUG_OUTPUT
              *out << "tgt_field_data, gold_value, abs_err, rel_err: "
                   << tgt_field_data[component] << ", " << gold_value[component] << ", " << abs_err_field_data[component]
                   << ", " << rel_err_field_data[component] << std::endl;
#endif
          }
        }

        // TARGET MESH WRITE
        // -----------------
        std::size_t tgt_output_index = tgt_broker.create_output_mesh(
            target_mesh_output_file, stk::io::WRITE_RESULTS );
        tgt_broker.add_field( tgt_output_index, target_interp_field );
        tgt_broker.add_field( tgt_output_index, target_rel_error_field );
        tgt_broker.add_field( tgt_output_index, target_abs_error_field );
        tgt_broker.add_field( tgt_output_index, *target_field );
        tgt_broker.begin_output_step( tgt_output_index, 0.0 );
        tgt_broker.write_defined_output_fields( tgt_output_index );
        tgt_broker.end_output_step( tgt_output_index );
        break; 
      }

      case 2: //TensorFieldType 
      { 
        typedef stk::mesh::Field<double, shards::ArrayDimension> TensorFieldType;
        //Get source_field from source mesh  
        TensorFieldType* source_field = src_broker.meta_data().get_field<TensorFieldType>(stk::topology::NODE_RANK, field_name); 
        if (source_field != 0) 
          *out << "Tensor Field with name " << field_name << " found in source mesh file!" << std::endl; 
        else   
          TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            std::endl << "Tensor Field with name " << field_name << " NOT found in source mesh file!" << std::endl); 

        int neq = source_field->max_size(stk::topology::NODE_RANK);
#ifdef DEBUG_OUTPUT
        *out << "Source field has " << neq << " dofs/node." << std::endl; 
#endif  

        //Put fields on target mesh 
        // Add a nodal field to the interpolated target part.
        TensorFieldType& target_interp_field = tgt_broker.meta_data().declare_field<TensorFieldType>( 
    	        stk::topology::NODE_RANK, tgt_interp_field_name );
        stk::mesh::put_field( target_interp_field, tgt_broker.meta_data().universal_part(), neq);
    
        // Add a absolute error nodal field to the target part.
        TensorFieldType& target_abs_error_field = tgt_broker.meta_data().declare_field<TensorFieldType>( 
    	    stk::topology::NODE_RANK, abs_err_field_name );
        stk::mesh::put_field( target_abs_error_field, tgt_broker.meta_data().universal_part(), neq);

        // Add a relative error nodal field to the target part.
        TensorFieldType& target_rel_error_field = tgt_broker.meta_data().declare_field<TensorFieldType>( 
    	    stk::topology::NODE_RANK, rel_err_field_name );
        stk::mesh::put_field( target_rel_error_field, tgt_broker.meta_data().universal_part(), neq);

        // Create the target bulk data.
        tgt_broker.populate_bulk_data();
        Teuchos::RCP<stk::mesh::BulkData> tgt_bulk_data = Teuchos::rcpFromRef( tgt_broker.bulk_data() );
    
        // Add a nodal field to the interpolated target part.
        //Populate target_field 
        TensorFieldType* target_field = tgt_broker.meta_data().get_field<TensorFieldType>(stk::topology::NODE_RANK, field_name); 
        if (target_field != 0) 
          *out << "Tensor Field with name " << field_name << " found in target mesh file!" << std::endl; 
        else   
          TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            std::endl << "Tensor Field with name " << field_name << " NOT found in target mesh file!" << std::endl); 
    
        io_region = tgt_broker.get_input_io_region();
        STKIORequire(!Teuchos::is_null(io_region));

        //Get number of time steps in source mesh 
        timestep_count = io_region->get_property("state_count").get_int();
#ifdef DEBUG_OUTPUT
        *out << "timestep_count in target mesh: " << timestep_count << std::endl;
#endif
        step =  tgt_snap_no; 
        if (step > timestep_count) 
          TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            std::endl << "Invalid value of Target Mesh Snapshot Number = " << tgt_snap_no <<
                         " > total number of snapshots in "  << source_mesh_input_file 
                      << " = " << timestep_count << "." << std::endl;);
        if (step <= 0) 
          TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            std::endl << "Invalid value of Target Mesh Snapshot Number = " << tgt_snap_no <<
                         "; value must be > 0." << std::endl;);  
        if (timestep_count > 0 ) { 
          double time = io_region->get_state_time(step);
          if (step == timestep_count)
            interpolation_intervals = 1;

          int step_end = step < timestep_count ? step+1 : step;
          double tend =  io_region->get_state_time(step_end);
          double tbeg = time;
          double delta = (tend - tbeg) / static_cast<double>(interpolation_intervals);

          for (int interval = 0; interval < interpolation_intervals; interval++) {
            time = tbeg + delta * static_cast<double>(interval);
            tgt_broker.read_defined_input_fields(time);
          }
        }

 
        // SOLUTION TRANSFER SETUP
        // -----------------------
    
        // Create a manager for the source part elements.
        DataTransferKit::STKMeshManager src_manager( src_bulk_data, src_stk_selector );

        // Create a manager for the target part nodes.
        stk::mesh::Selector tgt_stk_selector = stk::mesh::Selector(tgt_broker.meta_data().universal_part()); 
        DataTransferKit::STKMeshManager tgt_manager( tgt_bulk_data, tgt_stk_selector );

        // Create a solution vector for the source.
        Teuchos::RCP<Tpetra::MultiVector<double, int, DataTransferKit::SupportId>> src_vector =
          src_manager.createFieldMultiVector<TensorFieldType>(Teuchos::ptr(source_field), neq);

        // Create a solution vector for the target.
        Teuchos::RCP<Tpetra::MultiVector<double,int,DataTransferKit::SupportId> > tgt_vector =
	  tgt_manager.createFieldMultiVector<TensorFieldType>(
	  Teuchos::ptr(&target_interp_field), neq);

#ifdef DEBUG_OUTPUT
        // Print out source mesh info.
        Teuchos::RCP<Teuchos::Describable> src_describe = src_manager.functionSpace()->entitySet();
        *out << "Source Mesh: " << std::endl;
        src_describe->describe(*out, Teuchos::VERB_HIGH );
        *out << std::endl;

        // Print out target mesh info.
        Teuchos::RCP<Teuchos::Describable> tgt_describe = tgt_manager.functionSpace()->entitySet();
        *out << "Target Mesh: " << std::endl;
        tgt_describe->describe(*out, Teuchos::VERB_HIGH );
        *out << std::endl;
#endif

        // SOLUTION TRANSFER
        // -----------------

        // Create a map operator. The operator settings are in the
        // "DataTransferKit" parameter list.
        Teuchos::ParameterList& dtk_list = plist->sublist("DataTransferKit");    
        DataTransferKit::MapOperatorFactory op_factory;
        Teuchos::RCP<DataTransferKit::MapOperator> map_op = op_factory.create( src_vector->getMap(),
  	  		     tgt_vector->getMap(),
			     dtk_list );

        // Setup the map operator. This creates the underlying linear operators.
        map_op->setup( src_manager.functionSpace(), tgt_manager.functionSpace() );
    
        // Apply the map operator. This interpolates the data from one STK field
        // to the other.
        map_op->apply( *src_vector, *tgt_vector );
     
#ifdef DEBUG_OUTPUT
        *out << "src_vector: \n ";
        src_vector->describe(*out, Teuchos::VERB_EXTREME);
        *out << "tgt_vector: \n ";
        tgt_vector->describe(*out, Teuchos::VERB_EXTREME);
#endif
      
        // COMPUTE THE SOLUTION ERROR
        // --------------------------

        double* tgt_field_data;
        double* gold_value; //reference solution (i.e., target_interp_field) 
        double* rel_err_field_data;
        double* abs_err_field_data;
        std::vector< stk::mesh::Entity > tgt_ownednodes ;
        stk::mesh::Selector select_owned_in_part = stk::mesh::Selector( tgt_broker.meta_data().universal_part() ) &
                                               stk::mesh::Selector( tgt_broker.meta_data().locally_owned_part() );
        stk::mesh::get_selected_entities( select_owned_in_part ,
            tgt_broker.bulk_data().buckets( stk::topology::NODE_RANK ) ,
            tgt_ownednodes );
        int tgt_num_owned_nodes = tgt_ownednodes.size(); //number owned nodes
        stk::mesh::BucketVector tgt_part_buckets = tgt_stk_selector.get_buckets( stk::topology::NODE_RANK );
        std::vector<stk::mesh::Entity> tgt_part_nodes;
        stk::mesh::get_selected_entities(tgt_stk_selector, tgt_part_buckets, tgt_part_nodes );
        Intrepid::FieldContainer<double> tgt_node_coords = DataTransferKit::STKMeshHelpers::getEntityNodeCoordinates(
              Teuchos::Array<stk::mesh::Entity>(tgt_part_nodes), *tgt_bulk_data );
        int num_tgt_part_nodes = tgt_part_nodes.size(); //number nodes (owned + overlap) 
#ifdef DEBUG_OUTPUT
        std::cout << "proc #: " << comm->getRank() << ", tgt_num_owned_nodes = " << tgt_num_owned_nodes << std::endl; 
#endif

        double error_l2_norm_sq;
        double field_l2_norm_sq;
        for (int component = 0; component < neq; component++) {
          error_l2_norm_sq = 0.0; 
          field_l2_norm_sq = 0.0;
          
          for ( int n = 0; n < tgt_num_owned_nodes; ++n )
          {
            gold_value = stk::mesh::field_data( target_interp_field, tgt_ownednodes[n] );
            tgt_field_data = stk::mesh::field_data( *target_field, tgt_ownednodes[n] );
            rel_err_field_data = stk::mesh::field_data( target_rel_error_field, tgt_ownednodes[n] );
            rel_err_field_data[component] = std::abs(tgt_field_data[component] - gold_value[component]); 
            error_l2_norm_sq += rel_err_field_data[component] * rel_err_field_data[component];
            field_l2_norm_sq += tgt_field_data[component] * tgt_field_data[component];
          }
          *out << "Dof = " << component << std::endl;
          double error_l2_norm_global, field_l2_norm_global; 
          Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &error_l2_norm_sq, &error_l2_norm_global); 
          Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &field_l2_norm_sq, &field_l2_norm_global); 
          error_l2_norm_global = std::sqrt(error_l2_norm_global); 
          field_l2_norm_global = std::sqrt(field_l2_norm_global); 
          *out << "|e|_2 (abs error): " << error_l2_norm_global << std::endl; 
          *out << "|f|_2 (norm ref soln): " << field_l2_norm_global << std::endl; 
          *out << "|e|_2 / |f|_2 (rel error): " << error_l2_norm_global / field_l2_norm_global << std::endl;

          for ( int n = 0; n < num_tgt_part_nodes; ++n )
          {
            gold_value = stk::mesh::field_data( target_interp_field, tgt_part_nodes[n] );
            tgt_field_data = stk::mesh::field_data( *target_field, tgt_part_nodes[n] );
            rel_err_field_data = stk::mesh::field_data( target_rel_error_field, tgt_part_nodes[n] );
            abs_err_field_data = stk::mesh::field_data( target_abs_error_field, tgt_part_nodes[n] );
            rel_err_field_data[component] = std::abs(tgt_field_data[component] - gold_value[component]);
            abs_err_field_data[component] = std::abs(tgt_field_data[component] - gold_value[component]);
            if (gold_value[component] != 0)
              rel_err_field_data[component] /= std::abs(gold_value[component]);
#ifdef DEBUG_OUTPUT
              *out << "tgt_field_data, gold_value, abs_err, rel_err: "
                   << tgt_field_data[component] << ", " << gold_value[component] << ", " << abs_err_field_data[component]
                   << ", " << rel_err_field_data[component] << std::endl;
#endif
          }
        }

        // TARGET MESH WRITE
        // -----------------
        std::size_t tgt_output_index = tgt_broker.create_output_mesh(
            target_mesh_output_file, stk::io::WRITE_RESULTS );
        tgt_broker.add_field( tgt_output_index, target_interp_field );
        tgt_broker.add_field( tgt_output_index, target_rel_error_field );
        tgt_broker.add_field( tgt_output_index, target_abs_error_field );
        tgt_broker.add_field( tgt_output_index, *target_field );
        tgt_broker.begin_output_step( tgt_output_index, 0.0 );
        tgt_broker.write_defined_output_fields( tgt_output_index );
        tgt_broker.end_output_step( tgt_output_index );
        break; 
      }

    }

}

//---------------------------------------------------------------------------//
// end interpolation_error.cpp
//---------------------------------------------------------------------------//
