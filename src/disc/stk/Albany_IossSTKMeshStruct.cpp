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
#include <stk_mesh/fem/CreateAdjacentEntities.hpp>
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
                  const Teuchos::RCP<Teuchos::ParameterList>& params, bool adaptive,
		  const Teuchos::RCP<const Epetra_Comm>& comm) :
  GenericSTKMeshStruct(params),
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  useSerialMesh(false),
  adaptiveMesh(adaptive),
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
        print(*out, "Found side_rank entity:\n", *part);
         ssPartVec[part->name()]=part;
      }
    }
  }

  cullSubsetParts(ssNames, ssPartVec); // Eliminate sidesets that are subsets of other sidesets

#if 0
  // for debugging, print out the parts now
  std::map<std::string, stk::mesh::Part*>::iterator it;

  for(it = ssPartVec.begin(); it != ssPartVec.end(); ++it){ // loop over the parts in the map

    // for each part in turn, get the name of parts that are a subset of it

      print(*out, "Found \n", *it->second);
  }
  // end debugging
#endif

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

  // Restart index to read solution from exodus file.
  int index = params->get("Restart Index",-1); // Default to no restart
  double res_time = params->get<double>("Restart Time",-1.0); // Default to no restart

#ifdef ALBANY_ZOLTAN // rebalance needs Zoltan

  if(useSerialMesh){

    Ioss::Region *region = mesh_data->m_input_region;

  	bulkData->modification_begin();

    if(comm->MyPID() == 0){ // read in the mesh on PE 0

       stk::io::process_mesh_bulk_data(region, *bulkData);

      // Read solution from exodus file.
      if (index >= 0) { // User has specified a time step to restart at
        *out << "Restart Index set, reading solution index : " << index << endl;
         stk::io::input_mesh_fields(region, *bulkData, index);
         restartDataTime = region->get_state_time(index);
         hasRestartSolution = true;
      }
      else if (res_time >= 0) { // User has specified a time to restart at
        *out << "Restart solution time set, reading solution time : " << res_time << endl;
         stk::io::input_mesh_fields(region, *bulkData, res_time);
         restartDataTime = res_time;
         hasRestartSolution = true;
      }
      else {

        *out << "Neither restart index or time are set. Not reading solution data from exodus file"<< endl;

      }
    }

  	bulkData->modification_end();

  } // UseSerialMesh - reading mesh on PE 0

  else { // Parallel read from Nemspread files - or we are running in serial
#endif

    stk::io::populate_bulk_data(*bulkData, *mesh_data);

    // FIXME: This breaks sidesets and Neumann BC's. Something is removed from exo file?
    // this should only be called if we are doing adaptation as it adds overhead
    if(adaptiveMesh)

      addElementEdges();
#if 0
  // for debugging, print out the parts now

  std::map<std::string, stk::mesh::Part*>::iterator it;

  for(it = ssPartVec.begin(); it != ssPartVec.end(); ++it){ // loop over the parts in the map

    // for each part in turn, get the name of parts that are a subset of it
      print(*out, "After edges, Found \n\n", *it->second);
  }
  // end debugging
#endif


    if (!usePamgen)  {

      // Read solution from exodus file.
      if (index >= 0) { // User has specified a time step to restart at
        *out << "Restart Index set, reading solution index : " << index << endl;
         stk::io::process_input_request(*mesh_data, *bulkData, index);
         restartDataTime = region->get_state_time(index);
         hasRestartSolution = true;
      }
      else if (res_time >= 0) { // User has specified a time to restart at
        *out << "Restart solution time set, reading solution time : " << res_time << endl;
         stk::io::process_input_request(*mesh_data, *bulkData, res_time);
         restartDataTime = res_time;
         hasRestartSolution = true;
      }
      else {
        *out << "Restart Index not set. Not reading solution from exodus (" 
           << index << ")"<< endl;

      }
    }

    bulkData->modification_end();

#ifdef ALBANY_ZOLTAN
  } // Parallel Read - or running in serial
#endif

  if(hasRestartSolution){

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

void 
Albany::IossSTKMeshStruct::addElementEdges(){

// Add element edges (faces) to the mesh for topology adaptation
// From:  LCM/topology.cc

	stk::mesh::PartVector add_parts;
	stk::mesh::create_adjacent_entities(*(bulkData), add_parts);
// Note, if we return here sidesets are NOT broken!!!
return;
  stk::mesh::EntityRank elementRank = metaData->element_rank();
  stk::mesh::EntityRank nodeRank = metaData->node_rank();


	bulkData->modification_begin();

	std::vector<stk::mesh::Entity*> element_lst;
	stk::mesh::get_entities(*(bulkData),elementRank,element_lst);

// Somewhere here we are removing our sideset info!!! GAH

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
			if (j->entity_rank() != elementRank-1 && j->entity_rank() != nodeRank){
				del_relations.push_back(j->entity());
				del_ids.push_back(j->identifier());
			}
		}
		for (int j = 0; j < del_relations.size(); ++j){
			stk::mesh::Entity & entity = *(del_relations[j]);
			bulkData->destroy_relation(element,entity,del_ids[j]);
		}
	};

	if (elementRank == 3){
		// Remove extra relations from face
		std::vector<stk::mesh::Entity*> face_lst;
		stk::mesh::get_entities(*(bulkData),elementRank-1,face_lst);
		stk::mesh::EntityRank entityRank = face_lst[0]->entity_rank();
		for (int i = 0; i < face_lst.size(); ++i){
			stk::mesh::Entity & face = *(face_lst[i]);
			stk::mesh::PairIterRelation relations = face_lst[i]->relations();
			std::vector<stk::mesh::Entity*> del_relations;
			std::vector<int> del_ids;
			for (stk::mesh::PairIterRelation::iterator j = relations.begin();
					j != relations.end(); ++j){
				if (j->entity_rank() != entityRank+1 &&
						j->entity_rank() != entityRank-1){
					del_relations.push_back(j->entity());
					del_ids.push_back(j->identifier());
				}
			}
			for (int j = 0; j < del_relations.size(); ++j){
				stk::mesh::Entity & entity = *(del_relations[j]);
				bulkData->destroy_relation(face,entity,del_ids[j]);
			}
		}
	}


	bulkData->modification_end();

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
  validPL->set<double>("Restart Time", 1.0, "Exodus solution time to read for inital guess/condition.");
  validPL->set<bool>("Use Serial Mesh", false, "Read in a single mesh on PE 0 and rebalance");

  return validPL;
}
#endif
