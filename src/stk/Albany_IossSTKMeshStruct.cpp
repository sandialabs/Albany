//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifdef ALBANY_SEACAS

#include <iostream>

#include "Albany_IossSTKMeshStruct.hpp"
#include "Teuchos_VerboseObject.hpp"

#include <Shards_BasicTopologies.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/FieldData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_io/IossBridge.hpp>
#include <Ioss_SubSystem.h>

#include <stk_mesh/fem/FEMHelpers.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "Albany_Utils.hpp"

// Rebalance 
#ifdef ALBANY_ZOLTAN 
#include <stk_rebalance/Rebalance.hpp>
#include <stk_rebalance/Partition.hpp>
#include <stk_rebalance/ZoltanPartition.hpp>
#include <stk_rebalance_utils/RebalanceUtils.hpp>
#endif

Albany::IossSTKMeshStruct::IossSTKMeshStruct(
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
		  const Teuchos::RCP<const Epetra_Comm>& comm) :
  GenericSTKMeshStruct(params),
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  useSerialMesh(false),
  periodic(params->get("Periodic BC", false))
{
  params->validateParameters(*getValidDiscretizationParameters(),0);

  mesh_data = new stk::io::MeshData();

  usePamgen = (params->get("Method","Exodus") == "Pamgen");

#ifdef ALBANY_ZOLTAN  // rebalance requires Zoltan

  if (params->get<bool>("Use Serial Mesh", false) && comm->NumProc() > 1){ 
    // We are parallel but reading a single exodus file

    useSerialMesh = true;

    readSerialMesh(comm);

  }
  else 
#endif
    if (!usePamgen) {
    *out << "Albany_IOSS: Loading STKMesh from Exodus file  " 
         << params->get<string>("Exodus Input File Name") << endl;

    stk::io::create_input_mesh("exodusii",
                               params->get<string>("Exodus Input File Name"),
                               Albany::getMpiCommFromEpetraComm(*comm), 
                               *metaData, *mesh_data); 
    *out << "Albany_IOSS: Loading STKMesh from exodus file  " << endl;
  }
  else {
    *out << "Albany_IOSS: Loading STKMesh from Pamgen file  " 
         << params->get<string>("Pamgen Input File Name") << endl;

    stk::io::create_input_mesh("pamgen",
                               params->get<string>("Pamgen Input File Name"),
                               Albany::getMpiCommFromEpetraComm(*comm), 
                               *metaData, *mesh_data); 

  }

  numDim = metaData->spatial_dimension();

  stk::io::put_io_part_attribute(metaData->universal_part());

  // Set element blocks, side sets and node sets
  const stk::mesh::PartVector & all_parts = metaData->get_parts();
  std::vector<std::string> ssNames;
  std::vector<std::string> nsNames;
  int numEB = 0;

  for (stk::mesh::PartVector::const_iterator i = all_parts.begin();
       i != all_parts.end(); ++i) {

    stk::mesh::Part * const part = *i ;

    if ( part->primary_entity_rank() == metaData->element_rank()) {
      if (part->name()[0] != '{') {
      //*out << "IOSS-STK: Element part \"" << part->name() << "\" found " << endl;
         partVec[numEB] = part;
         numEB++;
      }
    }
    else if ( part->primary_entity_rank() == metaData->node_rank()) {
      if (part->name()[0] != '{') {
       //*out << "Mesh has Node Set ID: " << part->name() << endl;
         nsPartVec[part->name()]=part;
         nsNames.push_back(part->name());
      }
    }
    else if ( part->primary_entity_rank() == metaData->side_rank()) {
      if (part->name()[0] != '{') {
        //print(*out, "Found side_rank entity:\n", *part);
         ssPartVec[part->name()]=part;
//         ssNames.push_back(part->name());
      }
    }
  }

  cullSubsetParts(ssNames, ssPartVec); // Eliminate sidesets that are subsets of other sidesets

  int cub = params->get("Cubature Degree",3);
  int worksetSizeMax = params->get("Workset Size",50);

  // Get number of elements per element block using Ioss for use
  // in calculating an upper bound on the worksetSize.
  std::vector<int> el_blocks;
  stk::io::get_element_block_sizes(*mesh_data, el_blocks);
  TEUCHOS_TEST_FOR_EXCEPT(el_blocks.size() != partVec.size());

  int ebSizeMax =  *std::max_element(el_blocks.begin(), el_blocks.end());
  int worksetSize = this->computeWorksetSize(worksetSizeMax, ebSizeMax);

  // Build a map to get the EB name given the index

  for (int eb=0; eb<numEB; eb++) 

    this->ebNameToIndex[partVec[eb]->name()] = eb;

  // Construct MeshSpecsStruct
  if (!params->get("Separate Evaluators by Element Block",false)) {

    const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();
    this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub,
                               nsNames, ssNames, worksetSize, partVec[0]->name(), 
                               this->ebNameToIndex, this->interleavedOrdering));

  }
  else {

    *out << "MULTIPLE Elem Block in Ioss: DO worksetSize[eb] max?? " << endl; 
    this->allElementBlocksHaveSamePhysics=false;
    this->meshSpecs.resize(numEB);
    for (int eb=0; eb<numEB; eb++) {
      const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[eb]).getCellTopologyData();
      this->meshSpecs[eb] = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub,
                                                nsNames, ssNames, worksetSize, partVec[eb]->name(), 
                                                this->ebNameToIndex, this->interleavedOrdering));
      cout << "el_block_size[" << eb << "] = " << el_blocks[eb] << "   name  " << partVec[eb]->name() << endl; 
    }

  }

  {
    const Ioss::Region *inputRegion = mesh_data->m_input_region;
    this->solutionFieldHistoryDepth = inputRegion->get_property("state_count").get_int();
  }
}

Albany::IossSTKMeshStruct::~IossSTKMeshStruct()
{
  delete mesh_data;
}

void
Albany::IossSTKMeshStruct::readSerialMesh(const Teuchos::RCP<const Epetra_Comm>& comm){

#ifdef ALBANY_ZOLTAN // rebalance needs Zoltan

   MPI_Group group_world;
   MPI_Group peZero;
   MPI_Comm peZeroComm;

  // Read a single exodus mesh on Proc 0 then rebalance it across the machine

  MPI_Comm theComm = Albany::getMpiCommFromEpetraComm(*comm);

  int process_rank[1]; // the reader process

  process_rank[0] = 0;
  int my_rank;

  //get the group under theComm
  MPI_Comm_group(theComm, &group_world);
  // create the new group 
  MPI_Group_incl(group_world, 1, process_rank, &peZero);
  // create the new communicator 
  MPI_Comm_create(theComm, peZero, &peZeroComm);

  // Who am i?
  MPI_Comm_rank(peZeroComm, &my_rank);

  if(my_rank == 0){

    *out << "Albany_IOSS: Loading serial STKMesh from Exodus file  " 
         << params->get<string>("Exodus Input File Name") << endl;

  }

/* 
 * This checks the existence of the file, checks to see if we can open it, builds a handle to the region
 * and puts it in mesh_data (in_region), and reads the metaData into metaData.
 */

    stk::io::create_input_mesh("exodusii",
                               params->get<string>("Exodus Input File Name"), 
                               peZeroComm, 
                               *metaData, *mesh_data); 

  if(my_rank == 0){

    *out << "Albany_IOSS: Loading serial STKMesh from exodus file  " << endl;

  }

  // Here, all PEs have read the metaData from the input file, and have a pointer to in_region in mesh_data

#endif

}

void
Albany::IossSTKMeshStruct::setFieldAndBulkData(
                  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize)
{
  this->SetupFieldData(comm, neq_, sis, worksetSize);

  *out << "IOSS-STK: number of node sets = " << nsPartVec.size() << endl;
  *out << "IOSS-STK: number of side sets = " << ssPartVec.size() << endl;

  metaData->commit();

#ifdef ALBANY_ZOLTAN // rebalance needs Zoltan

  if(useSerialMesh){

    Ioss::Region *region = mesh_data->m_input_region;

  	bulkData->modification_begin();

    // Restart index to read solution from exodus file.
    int index = params->get("Restart Index",-1); // Default to no restart

    if(comm->MyPID() == 0){ // read in the mesh on PE 0

       readBulkData(*region);
// Comment the line above and uncomment the one below once changes checked into Trilinos stk::io catch up (GAH)
//       stk::io::process_mesh_bulk_data(region, *bulkData);

      // Read solution from exodus file.
      if (index<1) *out << "Restart Index not set. Not reading solution from exodus (" 
           << index << ")"<< endl;
      else {
        *out << "Restart Index set, reading solution time : " << index << endl;
// Uncomment the one below once changes checked into Trilinos stk::io catch up (GAH)
//         stk::io::input_mesh_fields(region, *bulkData, index);
      }

    }

    if (index >= 1)

         hasRestartSolution = true;

  	bulkData->modification_end();

  }
  else {
#endif

    stk::io::populate_bulk_data(*bulkData, *mesh_data);

    if (!usePamgen)  {
      // Restart index to read solution from exodus file.
      int index = params->get("Restart Index",-1); // Default to no restart
      if (index<1) *out << "Restart Index not set. Not reading solution from exodus (" 
           << index << ")"<< endl;
      else {
        *out << "Restart Index set, reading solution time : " << index << endl;
         Ioss::Region *region = mesh_data->m_input_region;
         stk::io::process_input_request(*mesh_data, *bulkData, index);
         hasRestartSolution = true;

         restartDataTime = region->get_state_time(index);
         Teuchos::Array<std::string> default_field;
         default_field.push_back("solution");
         Teuchos::Array<std::string> restart_fields =
           params->get<Teuchos::Array<std::string> >("Restart Fields", default_field);

		 // Get the fields to be used for restart

         // See what state data was initialized from the stk::io request
         // This should be propagated into stk::io
         const Ioss::ElementBlockContainer& elem_blocks = region->get_element_blocks();

/*
            // Uncomment to print what fields are in the exodus file
			Ioss::NameList exo_fld_names;
			elem_blocks[0]->field_describe(&exo_fld_names);
			for(std::size_t i = 0; i < exo_fld_names.size(); i++){
				*out << "Found field \"" << exo_fld_names[i] << "\" in exodus file" << std::endl;
			}
*/

         for (std::size_t i=0; i<sis->size(); i++) {
           Albany::StateStruct& st = *((*sis)[i]);

           if(elem_blocks[0]->field_exists(st.name))

             for(std::size_t j = 0; j < restart_fields.size(); j++)

               if(boost::iequals(st.name, restart_fields[j])){

                 *out << "Restarting from field \"" << st.name << "\" found in exodus file." << std::endl;
                 st.restartDataAvailable = true;
                 break;

               }
           }

      }
    }

    bulkData->modification_end();

#ifdef ALBANY_ZOLTAN
  }
#endif

  coordinates_field = metaData->get_field<VectorFieldType>(std::string("coordinates"));
  proc_rank_field = metaData->get_field<IntScalarFieldType>(std::string("proc_rank"));

  useElementAsTopRank = true;

#ifdef ALBANY_ZOLTAN
// Rebalance if we read a single mesh and are running in parallel

  if(useSerialMesh){

    double imbalance;

    stk::mesh::Selector selector(metaData->universal_part());
    stk::mesh::Selector owned_selector(metaData->locally_owned_part());

    imbalance = stk::rebalance::check_balance(*bulkData, NULL, 
      metaData->node_rank(), &selector);

    *out << "Before the rebalance, the imbalance threshold is = " << imbalance << endl;

    // Use Zoltan (default configuration) to determine new partition

    Teuchos::ParameterList emptyList;

    stk::rebalance::Zoltan zoltan_partition(Albany::getMpiCommFromEpetraComm(*comm), numDim, emptyList);

/*
    // Configure Zoltan to use graph-based partitioning

    Teuchos::ParameterList graph;
    Teuchos::ParameterList lb_method;
    lb_method.set("LOAD BALANCING METHOD", "4");
    graph.sublist(stk::rebalance::Zoltan::default_parameters_name()) = lb_method;

    stk::rebalance::Zoltan zoltan_partition(Albany::getMpiCommFromEpetraComm(*comm), numDim, graph);
*/

// Note that one has to use owned_selector below, unlike the rebalance use cases do (Why?)

    stk::rebalance::rebalance(*bulkData, owned_selector, coordinates_field, NULL, zoltan_partition);


    imbalance = stk::rebalance::check_balance(*bulkData, NULL, 
      metaData->node_rank(), &selector);

    *out << "After rebalancing, the imbalance threshold is = " << imbalance << endl;

  }
#endif

}

void
Albany::IossSTKMeshStruct::loadSolutionFieldHistory(int step)
{
  TEUCHOS_TEST_FOR_EXCEPT(step < 0 || step >= solutionFieldHistoryDepth);

  const int index = step + 1; // 1-based step indexing
  stk::io::process_input_request(*mesh_data, *bulkData, index);
}

/* 
 * GAH: TODO
 *
 * The function readBulkData duplicates the function stk::io::populate_bulk_data, with the exception of an
 * internal modification_begin() and modification_end(). When reading bulk data on a single processor (single
 * exodus file), all PEs must enter the modification_begin() / modification_end() block, but only one reads the
 * bulk data. TODO pull the modification statements from populate_bulk_data and retrofit.
 *
 */



void 
Albany::IossSTKMeshStruct::readBulkData(Ioss::Region& region){

  stk::mesh::BulkData& bulk = *bulkData;
  const stk::mesh::fem::FEMMetaData& fem_meta = *metaData;

  { // element blocks

    const Ioss::ElementBlockContainer& elem_blocks = region.get_element_blocks();
    for(Ioss::ElementBlockContainer::const_iterator it = elem_blocks.begin();
	it != elem_blocks.end(); ++it) {
      Ioss::ElementBlock *entity = *it;

      if (stk::io::include_entity(entity)) {
	const std::string &name = entity->name();
	stk::mesh::Part* const part = fem_meta.get_part(name);
	assert(part != NULL);

	const CellTopologyData* cell_topo = stk::io::get_cell_topology(*part);
	if (cell_topo == NULL) {
	  std::ostringstream msg ;
	  msg << " INTERNAL_ERROR: Part " << part->name() << " returned NULL from get_cell_topology()";
	  throw std::runtime_error( msg.str() );
	}

	std::vector<int> elem_ids ;
	std::vector<int> connectivity ;

	entity->get_field_data("ids", elem_ids);
	entity->get_field_data("connectivity", connectivity);

	size_t element_count = elem_ids.size();
	int nodes_per_elem = cell_topo->node_count ;

	std::vector<stk::mesh::EntityId> id_vec(nodes_per_elem);
	std::vector<stk::mesh::Entity*> elements(element_count);

	for(size_t i=0; i<element_count; ++i) {
	  int *conn = &connectivity[i*nodes_per_elem];
	  std::copy(&conn[0], &conn[0+nodes_per_elem], id_vec.begin());
	  elements[i] = &stk::mesh::fem::declare_element(bulk, *part, elem_ids[i], &id_vec[0]);
	}

	// Add all element attributes as fields.
	// If the only attribute is 'attribute', then add it; otherwise the other attributes are the
	// named components of the 'attribute' field, so add them instead.
	Ioss::NameList names;
	entity->field_describe(Ioss::Field::ATTRIBUTE, &names);
	for(Ioss::NameList::const_iterator I = names.begin(); I != names.end(); ++I) {
	  if(*I == "attribute" && names.size() > 1)
	    continue;
	  stk::mesh::FieldBase *field = fem_meta.get_field<stk::mesh::FieldBase> (*I);
	  if (field)
	    stk::io::field_data_from_ioss(field, elements, entity, *I);
	}
      }
    }
  }

  { // nodeblocks

    const Ioss::NodeBlockContainer& node_blocks = region.get_node_blocks();
    assert(node_blocks.size() == 1);

    Ioss::NodeBlock *nb = node_blocks[0];

    std::vector<stk::mesh::Entity*> nodes;
    stk::io::get_entity_list(nb, fem_meta.node_rank(), bulk, nodes);

    stk::mesh::Field<double,stk::mesh::Cartesian> *coord_field =
      fem_meta.get_field<stk::mesh::Field<double,stk::mesh::Cartesian> >("coordinates");

    stk::io::field_data_from_ioss(coord_field, nodes, nb, "mesh_model_coordinates");

  }

  { // nodesets

    const Ioss::NodeSetContainer& node_sets = region.get_nodesets();

    for(Ioss::NodeSetContainer::const_iterator it = node_sets.begin();
	it != node_sets.end(); ++it) {
      Ioss::NodeSet *entity = *it;

      if (stk::io::include_entity(entity)) {
	const std::string & name = entity->name();
	stk::mesh::Part* const part = fem_meta.get_part(name);
	assert(part != NULL);
	stk::mesh::PartVector add_parts( 1 , part );

	std::vector<int> node_ids ;
	int node_count = entity->get_field_data("ids", node_ids);

	std::vector<stk::mesh::Entity*> nodes(node_count);
	stk::mesh::EntityRank n_rank = fem_meta.node_rank();
	for(int i=0; i<node_count; ++i) {
	  nodes[i] = bulk.get_entity(n_rank, node_ids[i] );
	  if (nodes[i] != NULL)
	    bulk.declare_entity(n_rank, node_ids[i], add_parts );
	}

	stk::mesh::Field<double> *df_field =
	  fem_meta.get_field<stk::mesh::Field<double> >("distribution_factors");

	if (df_field != NULL) {
	  stk::io::field_data_from_ioss(df_field, nodes, entity, "distribution_factors");
	}
      }
    }
  }

  { // sidesets

    const Ioss::SideSetContainer& side_sets = region.get_sidesets();

    for(Ioss::SideSetContainer::const_iterator it = side_sets.begin();
	it != side_sets.end(); ++it) {
      Ioss::SideSet *entity = *it;

      if (stk::io::include_entity(entity)) {
//	process_surface_entity(entity, bulk);
  {
    assert(entity->type() == Ioss::SIDESET);

    size_t block_count = entity->block_count();
    for (size_t i=0; i < block_count; i++) {
      Ioss::SideBlock *block = entity->get_block(i);
      if (stk::io::include_entity(block)) {
	std::vector<int> side_ids ;
	std::vector<int> elem_side ;

	stk::mesh::Part * const sb_part = fem_meta.get_part(block->name());
	stk::mesh::EntityRank elem_rank = fem_meta.element_rank();

	block->get_field_data("ids", side_ids);
	block->get_field_data("element_side", elem_side);

	assert(side_ids.size() * 2 == elem_side.size());
	stk::mesh::PartVector add_parts( 1 , sb_part );

	size_t side_count = side_ids.size();
	std::vector<stk::mesh::Entity*> sides(side_count);
	for(size_t is=0; is<side_count; ++is) {
	  stk::mesh::Entity* const elem = bulk.get_entity(elem_rank, elem_side[is*2]);

	  // If NULL, then the element was probably assigned to an
	  // element block that appears in the database, but was
	  // subsetted out of the analysis mesh. Only process if
	  // non-null.
	  if (elem != NULL) {
	    // Ioss uses 1-based side ordinal, stk::mesh uses 0-based.
	    int side_ordinal = elem_side[is*2+1] - 1;

	    stk::mesh::Entity* side_ptr = NULL;
	    side_ptr = &stk::mesh::fem::declare_element_side(bulk, side_ids[is], *elem, side_ordinal);
	    stk::mesh::Entity& side = *side_ptr;

	    bulk.change_entity_parts( side, add_parts );
	    sides[is] = &side;
	  } else {
	    sides[is] = NULL;
	  }
	}

	const stk::mesh::Field<double, stk::mesh::ElementNode> *df_field =
	  stk::io::get_distribution_factor_field(*sb_part);
	if (df_field != NULL) {
	  stk::io::field_data_from_ioss(df_field, sides, block, "distribution_factors");
	}
      }
    }
  }

      }
    }
  }

}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::IossSTKMeshStruct::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getValidGenericSTKParameters("Valid IOSS_DiscParams");
  validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic a mesh");
  validPL->set<string>("Exodus Input File Name", "", "File Name For Exodus Mesh Input");
  validPL->set<string>("Pamgen Input File Name", "", "File Name For Pamgen Mesh Input");
  validPL->set<int>("Restart Index", 1, "Exodus time index to read for inital guess/condition.");
  validPL->set<bool>("Use Serial Mesh", false, "Read in a single mesh on PE 0 and rebalance");

  return validPL;
}
#endif
