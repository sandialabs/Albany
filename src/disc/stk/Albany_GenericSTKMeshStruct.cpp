//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>

#include "Albany_GenericSTKMeshStruct.hpp"


#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

#include "Albany_Utils.hpp"
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/fem/CreateAdjacentEntities.hpp>

// Rebalance 
#ifdef ALBANY_ZOLTAN
#include <stk_rebalance/Rebalance.hpp>
#include <stk_rebalance/Partition.hpp>
#include <stk_rebalance/ZoltanPartition.hpp>
#include <stk_rebalance_utils/RebalanceUtils.hpp>
#endif

// Refinement
//#include <stk_percept/PerceptMesh.hpp>
//#include <stk_adapt/UniformRefiner.hpp>

Albany::GenericSTKMeshStruct::GenericSTKMeshStruct(
    const Teuchos::RCP<Teuchos::ParameterList>& params_,
    int numDim_)
    : params(params_)
{
  metaData = new stk::mesh::fem::FEMMetaData();
  
  // numDim = -1 is default flag value to postpone initialization
  if (numDim_>0) {
    this->numDim = numDim_;
    metaData->FEM_initialize(numDim_);
  }

  interleavedOrdering = params->get("Interleaved Ordering",true);
  allElementBlocksHaveSamePhysics = true; 
  hasRestartSolution = false;

  // No history available by default
  solutionFieldHistoryDepth = 0;

  // This is typical, can be resized for multiple material problems
  meshSpecs.resize(1);

  bulkData = NULL;
}

Albany::GenericSTKMeshStruct::~GenericSTKMeshStruct()
{
  delete metaData;
  delete bulkData;
}

void Albany::GenericSTKMeshStruct::SetupFieldData(
		  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const int neq_,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const int worksetSize) 
{
  TEUCHOS_TEST_FOR_EXCEPTION(!metaData->is_FEM_initialized(),
       std::logic_error,
       "LogicError: metaData->FEM_initialize(numDim) not yet called" << std::endl);

  neq = neq_;

  if (bulkData ==  NULL)
  bulkData = new stk::mesh::BulkData(stk::mesh::fem::FEMMetaData::get_meta_data(*metaData),
                          Albany::getMpiCommFromEpetraComm(*comm), worksetSize );

  //Start STK stuff
  coordinates_field = & metaData->declare_field< VectorFieldType >( "coordinates" );
  proc_rank_field = & metaData->declare_field< IntScalarFieldType >( "proc_rank" );
  solution_field = & metaData->declare_field< VectorFieldType >(
    params->get<string>("Exodus Solution Name", "solution"));
#ifdef ALBANY_LCM
  residual_field = & metaData->declare_field< VectorFieldType >(
    params->get<string>("Exodus Residual Name", "residual"));

  // Build fields to indicate open entities for topology modification
/*
  faces_open_field = & metaData->declare_field< BoolScalarFieldType >(
    params->get<string>("Faces open field", "faces_open"));
  segments_open_field = & metaData->declare_field< BoolScalarFieldType >(
    params->get<string>("Segments open field", "segments_open"));
  nodes_open_field = & metaData->declare_field< BoolScalarFieldType >(
    params->get<string>("Nodes open field", "nodes_open"));
*/
#endif

  stk::mesh::put_field( *coordinates_field , metaData->node_rank() , metaData->universal_part(), numDim );
  // Processor rank field, a scalar
  stk::mesh::put_field( *proc_rank_field , metaData->element_rank() , metaData->universal_part());
  stk::mesh::put_field( *solution_field , metaData->node_rank() , metaData->universal_part(), neq );
#ifdef ALBANY_LCM
  stk::mesh::put_field( *residual_field , metaData->node_rank() , metaData->universal_part() , neq );

  // Build fields to indicate open entities for topology modification
/*
  stk::mesh::put_field( *faces_open_field , metaData->element_rank() - 1 , metaData->universal_part());
  stk::mesh::put_field( *segments_open_field , metaData->node_rank() + 1 , metaData->universal_part());
  stk::mesh::put_field( *nodes_open_field , metaData->node_rank(), metaData->universal_part());
*/
#endif
  
#ifdef ALBANY_SEACAS
  stk::io::set_field_role(*coordinates_field, Ioss::Field::MESH);
  stk::io::set_field_role(*proc_rank_field, Ioss::Field::MESH);
  stk::io::set_field_role(*solution_field, Ioss::Field::TRANSIENT);
#ifdef ALBANY_LCM
  stk::io::set_field_role(*residual_field, Ioss::Field::TRANSIENT);

/*
  stk::io::set_field_role(*faces_open_field, Ioss::Field::MESH);
  stk::io::set_field_role(*segments_open_field, Ioss::Field::MESH);
  stk::io::set_field_role(*nodes_open_field, Ioss::Field::MESH);
*/
#endif
#endif

  // Code to parse the vector of StateStructs and create STK fields
  for (std::size_t i=0; i<sis->size(); i++) {
    Albany::StateStruct& st = *((*sis)[i]);
    std::vector<int>& dim = st.dim;
    if (dim.size() == 2 && st.entity=="QuadPoint") {
      qpscalar_states.push_back(& metaData->declare_field< QPScalarFieldType >( st.name) );
      stk::mesh::put_field( *qpscalar_states.back() , metaData->element_rank(),
			    metaData->universal_part(), dim[1]);
      cout << "NNNN qps field name " << qpscalar_states.back()->name() << endl;
#ifdef ALBANY_SEACAS
      if (st.output) stk::io::set_field_role(*qpscalar_states.back(), Ioss::Field::TRANSIENT);
#endif
    }
    else if (dim.size() == 3 && st.entity=="QuadPoint") {
      qpvector_states.push_back(& metaData->declare_field< QPVectorFieldType >( st.name) );
      // Multi-dim order is Fortran Ordering, so reversed here
      stk::mesh::put_field( *qpvector_states.back() , metaData->element_rank(),
			    metaData->universal_part(), dim[2], dim[1]);
      cout << "NNNN qpv field name " << qpvector_states.back()->name() << endl;
#ifdef ALBANY_SEACAS
      if (st.output) stk::io::set_field_role(*qpvector_states.back(), Ioss::Field::TRANSIENT);
#endif
    }
    else if (dim.size() == 4 && st.entity=="QuadPoint") {
      qptensor_states.push_back(& metaData->declare_field< QPTensorFieldType >( st.name) );
      // Multi-dim order is Fortran Ordering, so reversed here
      stk::mesh::put_field( *qptensor_states.back() , metaData->element_rank(),
			    metaData->universal_part(), dim[3], dim[2], dim[1]);
      cout << "NNNN qpt field name " << qptensor_states.back()->name() << endl;
#ifdef ALBANY_SEACAS
      if (st.output) stk::io::set_field_role(*qptensor_states.back(), Ioss::Field::TRANSIENT);
#endif
    }
    else if ( dim.size() == 1 && st.entity=="ScalarValue" ) {
      scalarValue_states.push_back(st.name);
    }
    else TEUCHOS_TEST_FOR_EXCEPT(dim.size() < 2 || dim.size()>4 || st.entity!="QuadPoint");

  }
  
  // Exodus is only for 2D and 3D. Have 1D version as well
  exoOutput = params->isType<string>("Exodus Output File Name");
  if (exoOutput)
    exoOutFile = params->get<string>("Exodus Output File Name");

  exoOutputInterval = params->get<int>("Exodus Write Interval", 1);
  
  
  //get the type of transformation of STK mesh (for FELIX problems)
  transformType = params->get("Transform Type", "None"); //get the type of transformation of STK mesh (for FELIX problems)
  felixAlpha = params->get("FELIX alpha", 0.0); 
  felixL = params->get("FELIX L", 1.0); 
}

void Albany::GenericSTKMeshStruct::DeclareParts(std::vector<std::string> ebNames, std::vector<std::string> ssNames,
  std::vector<std::string> nsNames)
{
  // Element blocks
  for (std::size_t i=0; i<ebNames.size(); i++) {
    std::string ebn = ebNames[i];
    partVec[i] = & metaData->declare_part(ebn, metaData->element_rank() );
#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*partVec[i]);
#endif
  }

  // SideSets
  for (std::size_t i=0; i<ssNames.size(); i++) {
    std::string ssn = ssNames[i];
    ssPartVec[ssn] = & metaData->declare_part(ssn, metaData->side_rank() );
#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*ssPartVec[ssn]);
#endif
  }

  // NodeSets
  for (std::size_t i=0; i<nsNames.size(); i++) {
    std::string nsn = nsNames[i];
    nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  }
}

void 
Albany::GenericSTKMeshStruct::cullSubsetParts(std::vector<std::string>& ssNames, 
    std::map<std::string, stk::mesh::Part*>& partVec){

/*
When dealing with sideset lists, it is common to have parts that are subsets of other parts, like:
Part[ surface_12 , 18 ] {
  Supersets { {UNIVERSAL} }
  Intersection_Of { } }
  Subsets { surface_quad4_edge2d2_12 }

Part[ surface_quad4_edge2d2_12 , 19 ] {
  Supersets { {UNIVERSAL} {FEM_ROOT_CELL_TOPOLOGY_PART_Line_2} surface_12 }
  Intersection_Of { } }
  Subsets { }

This function gets rid of the subset in the list. 
*/

  using std::map;

  map<std::string, stk::mesh::Part*>::iterator it;
  std::vector<stk::mesh::Part*>::const_iterator p;

  for(it = partVec.begin(); it != partVec.end(); ++it){ // loop over the parts in the map

    // for each part in turn, get the name of parts that are a subset of it

    const stk::mesh::PartVector & subsets   = it->second->subsets();

    for ( p = subsets.begin() ; p != subsets.end() ; ++p ) {
      const std::string & n = (*p)->name();
//      std::cout << "Erasing: " << n << std::endl;
      partVec.erase(n); // erase it if it is in the base map
    }
  }

//  ssNames.clear();

  // Build the remaining data structures
  for(it = partVec.begin(); it != partVec.end(); ++it){ // loop over the parts in the map

    std::string ssn = it->first;
    ssNames.push_back(ssn);

  }
}


Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >&
Albany::GenericSTKMeshStruct::getMeshSpecs()
{
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs==Teuchos::null,
       std::logic_error,
       "meshSpecs accessed, but it has not been constructed" << std::endl);
  return meshSpecs;
}

int Albany::GenericSTKMeshStruct::computeWorksetSize(const int worksetSizeMax,
                                                     const int ebSizeMax) const
{
  // Resize workset size down to maximum number in an element block
  if (worksetSizeMax > ebSizeMax || worksetSizeMax < 1) return ebSizeMax;
  else {
     // compute numWorksets, and shrink workset size to minimize padding
     const int numWorksets = 1 + (ebSizeMax-1) / worksetSizeMax;
     return (1 + (ebSizeMax-1) /  numWorksets);
  }
}

void Albany::GenericSTKMeshStruct::computeAddlConnectivity()
{

  stk::mesh::PartVector add_parts;
  stk::mesh::create_adjacent_entities(*bulkData, add_parts);

  stk::mesh::EntityRank elementRank = metaData->element_rank();
  stk::mesh::EntityRank nodeRank = metaData->node_rank();
  stk::mesh::EntityRank sideRank = metaData->side_rank();

  std::vector<stk::mesh::Entity*> element_lst;
//  stk::mesh::get_entities(*(bulkData),elementRank,element_lst);
  
  stk::mesh::Selector select_owned_or_shared = metaData->locally_owned_part() | metaData->globally_shared_part();
  stk::mesh::Selector select_owned = metaData->locally_owned_part();

/*
      stk::mesh::Selector select_owned_in_part =
      stk::mesh::Selector( metaData->universal_part() ) &
      stk::mesh::Selector( metaData->locally_owned_part() );

      stk::mesh::get_selected_entities( select_owned_in_part ,
*/

   // Loop through only on-processor elements as we are just deleting entities inside the element
   stk::mesh::get_selected_entities( select_owned,
      bulkData->buckets( elementRank ) ,
      element_lst );

  bulkData->modification_begin();

    // Remove extra relations from element
  for (int i = 0; i < element_lst.size(); ++i){
      stk::mesh::Entity & element = *(element_lst[i]);
      stk::mesh::PairIterRelation relations = element.relations();
      std::vector<stk::mesh::Entity*> del_relations;
      std::vector<int> del_ids;
      for (stk::mesh::PairIterRelation::iterator j = relations.begin();
           j != relations.end(); ++j){

        // remove all relationships from element unless to faces(segments
        //   in 2D) or nodes 

        if (
            j->entity_rank() != elementRank-1 && // element to face relation
            j->entity_rank() != nodeRank  
           ){

          del_relations.push_back(j->entity());
          del_ids.push_back(j->identifier());
        }
      }

    for (int j = 0; j < del_relations.size(); ++j){
      stk::mesh::Entity & entity = *(del_relations[j]);
      bulkData->destroy_relation(element,entity,del_ids[j]);
    }
  }

  if (elementRank == 3){
    // Remove extra relations from face
    std::vector<stk::mesh::Entity*> face_lst;
    //stk::mesh::get_entities(*(bulkData),elementRank-1,face_lst);
    // Loop through all faces visible to this processor, as a face can be visible on two processors
    stk::mesh::get_selected_entities( select_owned_or_shared,
                                      bulkData->buckets( elementRank-1 ) ,
                                      face_lst );
    stk::mesh::EntityRank entityRank = face_lst[0]->entity_rank(); // This is rank 2 always...
//std::cout << "element rank - 1: " << elementRank - 1 << " face rank: " << entityRank << std::endl;
    for (int i = 0; i < face_lst.size(); ++i){
      stk::mesh::Entity & face = *(face_lst[i]);
      stk::mesh::PairIterRelation relations = face.relations();
      std::vector<stk::mesh::Entity*> del_relations;
      std::vector<int> del_ids;
      for (stk::mesh::PairIterRelation::iterator j = relations.begin();
           j != relations.end(); ++j){

        if (
            j->entity_rank() != entityRank+1 && // face to element relation
            j->entity_rank() != entityRank-1 // && // face to segment relation
//            j->entity_rank() != sideRank     ){
           ){

          del_relations.push_back(j->entity());
          del_ids.push_back(j->identifier());
        }
      }

      for (int j = 0; j < del_relations.size(); ++j){
        stk::mesh::Entity & entity = *(del_relations[j]);
        bulkData->destroy_relation(face, entity, del_ids[j]);
//std::cout << "Deleting rank: " << entity.entity_rank() << " id: " << del_ids[j] << std::endl;
      }
    }
  }

  bulkData->modification_end();

}

void Albany::GenericSTKMeshStruct::uniformRefineMesh(const Teuchos::RCP<const Epetra_Comm>& comm){

#if 0
// Refine if requested

    stk::percept::PerceptMesh eMesh(metaData, bulkData, false);
//    eMesh.printInfo("Mesh input to refiner", 0);

  // Reopen the metaData so we can refine the existing mesh
    eMesh.reopen();
    stk::adapt::Quad4_Quad4_4 subdivideQuads(eMesh);
    eMesh.commit();

    stk::adapt::UniformRefiner refiner(eMesh, subdivideQuads, proc_rank_field);

    refiner.doBreak();
#endif

}


void Albany::GenericSTKMeshStruct::rebalanceMesh(const Teuchos::RCP<const Epetra_Comm>& comm){

// Zoltan is required here

#ifdef ALBANY_ZOLTAN

    double imbalance;

    stk::mesh::Selector selector(metaData->universal_part());
    stk::mesh::Selector owned_selector(metaData->locally_owned_part());

    cout << "Before rebal nelements " << comm->MyPID() << "  " << 
      stk::mesh::count_selected_entities(owned_selector, bulkData->buckets(metaData->element_rank())) << endl;

    cout << "Before rebal " << comm->MyPID() << "  " << 
      stk::mesh::count_selected_entities(owned_selector, bulkData->buckets(metaData->node_rank())) << endl;


    imbalance = stk::rebalance::check_balance(*bulkData, NULL, 
      metaData->node_rank(), &selector);

    if(comm->MyPID() == 0){

      cout << "Before first rebal: Imbalance threshold is = " << imbalance << endl;

    }

    // Use Zoltan to determine new partition
    Teuchos::ParameterList emptyList;

    stk::rebalance::Zoltan zoltan_partition(Albany::getMpiCommFromEpetraComm(*comm), numDim, emptyList);
    stk::rebalance::rebalance(*bulkData, owned_selector, coordinates_field, NULL, zoltan_partition);


    imbalance = stk::rebalance::check_balance(*bulkData, NULL, 
      metaData->node_rank(), &selector);

    if(comm->MyPID() == 0){

      cout << "Before second rebal: Imbalance threshold is = " << imbalance << endl;

    }


    // Configure Zoltan to use graph-based partitioning
    Teuchos::ParameterList graph;
    Teuchos::ParameterList lb_method;
    lb_method.set("LOAD BALANCING METHOD"      , "4");
    graph.sublist(stk::rebalance::Zoltan::default_parameters_name()) = lb_method;

    stk::rebalance::Zoltan zoltan_partitiona(Albany::getMpiCommFromEpetraComm(*comm), numDim, graph);

    cout << "Universal part " << comm->MyPID() << "  " << 
      stk::mesh::count_selected_entities(selector, bulkData->buckets(metaData->element_rank())) << endl;
    cout << "Owned part " << comm->MyPID() << "  " << 
      stk::mesh::count_selected_entities(owned_selector, bulkData->buckets(metaData->element_rank())) << endl;

    stk::rebalance::rebalance(*bulkData, owned_selector, coordinates_field, NULL, zoltan_partitiona);

    cout << "After rebal " << comm->MyPID() << "  " << 
      stk::mesh::count_selected_entities(owned_selector, bulkData->buckets(metaData->node_rank())) << endl;
    cout << "After rebal nelements " << comm->MyPID() << "  " << 
      stk::mesh::count_selected_entities(owned_selector, bulkData->buckets(metaData->element_rank())) << endl;


    imbalance = stk::rebalance::check_balance(*bulkData, NULL, 
      metaData->node_rank(), &selector);

    if(comm->MyPID() == 0){

      cout << "Before second rebal: Imbalance threshold is = " << imbalance << endl;

    }

#endif  //ALBANY_ZOLTAN

}

Teuchos::RCP<Teuchos::ParameterList>
Albany::GenericSTKMeshStruct::getValidGenericSTKParameters(std::string listname) const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList(listname));;
  validPL->set<string>("Cell Topology", "Quad" , "Quad or Tri Cell Topology");
  validPL->set<std::string>("Exodus Output File Name", "",
      "Request exodus output to given file name. Requires SEACAS build");
  validPL->set<std::string>("Exodus Solution Name", "",
      "Name of solution output vector written to Exodus file. Requires SEACAS build");
  validPL->set<std::string>("Exodus Residual Name", "",
      "Name of residual output vector written to Exodus file. Requires SEACAS build");
  validPL->set<int>("Exodus Write Interval", 3, "Step interval to write solution data to Exodus file");
  validPL->set<std::string>("Method", "",
    "The discretization method, parsed in the Discretization Factory");
  validPL->set<int>("Cubature Degree", 3, "Integration order sent to Intrepid");
  validPL->set<int>("Workset Size", 50, "Upper bound on workset (bucket) size");
  validPL->set<bool>("Interleaved Ordering", true, "Flag for interleaved or blocked unknown ordering");
  validPL->set<bool>("Separate Evaluators by Element Block", false,
                     "Flag for different evaluation trees for each Element Block");
  validPL->set<string>("Transform Type", "None", "None or ISMIP-HOM Test A"); //for FELIX problem that require tranformation of STK mesh
  validPL->set<double>("FELIX alpha", 0.0, "Surface boundary inclination for FELIX problems (in degrees)"); //for FELIX problem that require tranformation of STK mesh
  validPL->set<double>("FELIX L", 1, "Domain length for FELIX problems"); //for FELIX problem that require tranformation of STK mesh
  Teuchos::Array<std::string> defaultFields;
  validPL->set<Teuchos::Array<std::string> >("Restart Fields", defaultFields, 
                     "Fields to pick up from the restart file when restarting");


  return validPL;
}
