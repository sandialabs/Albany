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
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_io/IossBridge.hpp>
#include <Ioss_SubSystem.h>

//#include <stk_mesh/fem/FEMHelpers.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "Albany_Utils.hpp"

namespace {

void get_element_block_sizes(stk::io::StkMeshIoBroker &mesh_data,
                             std::vector<int>& el_blocks)
{
  Ioss::Region &io = *mesh_data.get_input_io_region();
  const Ioss::ElementBlockContainer& elem_blocks = io.get_element_blocks();
  for(Ioss::ElementBlockContainer::const_iterator it = elem_blocks.begin(); it != elem_blocks.end(); ++it) {
    Ioss::ElementBlock *entity = *it;
    if (stk::io::include_entity(entity)) {
      el_blocks.push_back(entity->get_property("entity_count").get_int());
    }
  }
}

}

Albany::IossSTKMeshStruct::IossSTKMeshStruct(
                                             const Teuchos::RCP<Teuchos::ParameterList>& params,
                                             const Teuchos::RCP<Teuchos::ParameterList>& adaptParams_,
                                             const Teuchos::RCP<const Teuchos_Comm>& commT) :
  GenericSTKMeshStruct(params, adaptParams_),
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  useSerialMesh(false),
  periodic(params->get("Periodic BC", false)),
  m_hasRestartSolution(false),
  m_restartDataTime(-1.0),
  m_solutionFieldHistoryDepth(0)
{
  params->validateParameters(*getValidDiscretizationParameters(),0);

  usePamgen = (params->get("Method","Exodus") == "Pamgen");


  const Teuchos::MpiComm<int>* mpiComm = dynamic_cast<const Teuchos::MpiComm<int>* > (commT.get());
  std::vector<std::string> entity_rank_names = stk::mesh::entity_rank_names();

  // eMesh needs "FAMILY_TREE" entity
  if(buildEMesh) {
    entity_rank_names.push_back("FAMILY_TREE");
  }

  const Teuchos::MpiComm<int>* theComm = dynamic_cast<const Teuchos::MpiComm<int>* > (commT.get());
  if (params->get<bool>("Use Serial Mesh", false) && commT->getSize() > 1){
    // We are parallel but reading a single exodus file
    useSerialMesh = true;

    // Read a single exodus mesh on Proc 0 then rebalance it across the machine
    MPI_Group group_world;
    MPI_Group peZero;
    MPI_Comm peZeroComm;
    //MPI_Comm theComm = Albany::getMpiCommFromEpetraComm(*comm);
    int process_rank[1]; // the reader process
    process_rank[0] = 0;
    int my_rank = commT->getRank();

    //get the group under theComm
    MPI_Comm_group(*theComm->getRawMpiComm(), &group_world);
    // create the new group. This group includes only processor zero - that is the only processor that reads the file
    MPI_Group_incl(group_world, 1, process_rank, &peZero);
    // create the new communicator - it just contains processor zero
    MPI_Comm_create(*theComm->getRawMpiComm(), peZero, &peZeroComm);

    mesh_data = Teuchos::rcp(new stk::io::StkMeshIoBroker(peZeroComm));
  }
  else {
    mesh_data = Teuchos::rcp(new stk::io::StkMeshIoBroker(*theComm->getRawMpiComm()));
  }

  // Create input mesh

  mesh_data->set_rank_name_vector(entity_rank_names);
  mesh_data->set_sideset_face_creation_behavior(stk::io::StkMeshIoBroker::STK_IO_SIDESET_FACE_CREATION_CLASSIC);
  //StkMeshIoBroker::set_sideset_face_creation_behavior(stk::io::StkMeshIoBroker::STK_IO_SIDESET_FACE_CREATION_CLASSIC);
  std::string mesh_type;
  std::string file_name;
  if (!usePamgen) {
    *out << "Albany_IOSS: Loading STKMesh from Exodus file  "
         << params->get<std::string>("Exodus Input File Name") << std::endl;

    mesh_type = "exodusII";
    file_name = params->get<std::string>("Exodus Input File Name");
  }
  else {
    *out << "Albany_IOSS: Loading STKMesh from Pamgen file  "
         << params->get<std::string>("Pamgen Input File Name") << std::endl;

    mesh_type = "pamgen";
    file_name = params->get<std::string>("Pamgen Input File Name");
  }

  mesh_data->add_mesh_database(file_name, mesh_type, stk::io::READ_MESH);
  mesh_data->create_input_mesh();

  metaData = mesh_data->meta_data_rcp();//Teuchos::rcpFromRef(mesh_data->meta_data());

  // End of creating input mesh

  typedef Teuchos::Array<std::string> StringArray;
  const StringArray additionalNodeSets = params->get("Additional Node Sets", StringArray());
  for (StringArray::const_iterator it = additionalNodeSets.begin(), it_end = additionalNodeSets.end(); it != it_end; ++it) {
    stk::mesh::Part &newNodeSet = metaData->declare_part(*it, stk::topology::NODE_RANK);
    if (!stk::io::is_part_io_part(newNodeSet)) {
      stk::mesh::Field<double> * const distrFactorfield = metaData->get_field<stk::mesh::Field<double> >(stk::topology::NODE_RANK, "distribution_factors");
      stk::mesh::put_field(*distrFactorfield, newNodeSet);
      stk::io::put_io_part_attribute(newNodeSet);
    }
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

    if (!stk::mesh::is_auto_declared_part(*part)) {
      if ( part->primary_entity_rank() == stk::topology::ELEMENT_RANK) {

        //*out << "IOSS-STK: Element part \"" << part->name() << "\" found " << std::endl;
        partVec[numEB] = part;
        numEB++;
      }
      else if ( part->primary_entity_rank() == stk::topology::NODE_RANK) {
        //*out << "Mesh has Node Set ID: " << part->name() << std::endl;
        nsPartVec[part->name()]=part;
        nsNames.push_back(part->name());
      }
      else if ( part->primary_entity_rank() == metaData->side_rank()) {
        //print(*out, "Found side_rank entity:\n", *part);
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

  const int cub      = params->get("Cubature Degree",3);
  const int default_cub_rule = static_cast<int>(Intrepid::PL_GAUSS);
  const Intrepid::EIntrepidPLPoly cub_rule = static_cast<Intrepid::EIntrepidPLPoly>(params->get("Cubature Rule",default_cub_rule));
  int worksetSizeMax = params->get("Workset Size",50);

  // Get number of elements per element block using Ioss for use
  // in calculating an upper bound on the worksetSize.
  std::vector<int> el_blocks;
  get_element_block_sizes(*mesh_data, el_blocks);
  TEUCHOS_TEST_FOR_EXCEPT(el_blocks.size() != partVec.size());

  int ebSizeMax =  *std::max_element(el_blocks.begin(), el_blocks.end());
  int worksetSize = this->computeWorksetSize(worksetSizeMax, ebSizeMax);

  // Build a map to get the EB name given the index

  for (int eb=0; eb<numEB; eb++)

    this->ebNameToIndex[partVec[eb]->name()] = eb;

  // Construct MeshSpecsStruct
  if (!params->get("Separate Evaluators by Element Block",false)) {

    const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();
    this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(
        ctd, numDim, cub, nsNames, ssNames, worksetSize, partVec[0]->name(),
        this->ebNameToIndex, this->interleavedOrdering, false, cub_rule));

  }
  else {

    *out << "MULTIPLE Elem Block in Ioss: DO worksetSize[eb] max?? " << std::endl;
    this->allElementBlocksHaveSamePhysics=false;
    this->meshSpecs.resize(numEB);
    for (int eb=0; eb<numEB; eb++) {
      const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[eb]).getCellTopologyData();
      this->meshSpecs[eb] = Teuchos::rcp(new Albany::MeshSpecsStruct(
          ctd, numDim, cub, nsNames, ssNames, worksetSize, partVec[eb]->name(),
          this->ebNameToIndex, this->interleavedOrdering, true, cub_rule));
      //std::cout << "el_block_size[" << eb << "] = " << el_blocks[eb] << "   name  " << partVec[eb]->name() << std::endl;
    }

  }

  {
    const Ioss::Region& inputRegion = *(mesh_data->get_input_io_region());
    m_solutionFieldHistoryDepth = inputRegion.get_property("state_count").get_int();
  }
}

Albany::IossSTKMeshStruct::~IossSTKMeshStruct()
{
  // Explicitly delete these in exactly this order. There are three
  // reasons. First, StkMeshIoBroker does not have a MetaData getter that
  // returns a Teuchos::RCP. Second, bulkData is constructed with a raw ref to
  // metaData. Third, bulkData uses metaData in its destructor.
  //   If these are not set to null in the following order, mesh_data will be
  // deleted here, and then later bulkData and metaData will be deleted in
  // AbstractMeshStruct. But mesh_data's destructor will invalidate metaData,
  // and so there is then a memory problem when bulkData is deleted.
  //   I recommend that the following methods be added to STK:
  //     Teuchos::RCP<stk::mesh::MetaData> meta_data();
  //     BulkData(const Teuchos::RCP<mesh_meta_data>, ...);
  // Until then, Albany needs to be careful with these three objects.
  bulkData = Teuchos::null;
  metaData = Teuchos::null;
  mesh_data = Teuchos::null;
}

void
Albany::IossSTKMeshStruct::setFieldAndBulkData (
          const Teuchos::RCP<const Teuchos_Comm>& commT,
          const Teuchos::RCP<Teuchos::ParameterList>& params,
          const unsigned int neq_,
          const AbstractFieldContainer::FieldContainerRequirements& req,
          const Teuchos::RCP<Albany::StateInfoStruct>& sis,
          const unsigned int worksetSize,
          const Teuchos::RCP<std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> > >& ss_/*sis*/)
{
  this->SetupFieldData(commT, neq_, req, sis, worksetSize);

  mesh_data->set_bulk_data(*bulkData);

  *out << "IOSS-STK: number of node sets = " << nsPartVec.size() << std::endl;
  *out << "IOSS-STK: number of side sets = " << ssPartVec.size() << std::endl;

  mesh_data->add_all_mesh_fields_as_input_fields();
  std::vector<stk::io::MeshField> missing;

  metaData->commit();

  // Restart index to read solution from exodus file.
  int index = params->get("Restart Index",-1); // Default to no restart
  double res_time = params->get<double>("Restart Time",-1.0); // Default to no restart
  Ioss::Region& region = *(mesh_data->get_input_io_region());

  /*
   * The following code block reads a single mesh on PE 0, then distributes the mesh across
   * the other processors. stk_rebalance is used, which requires Zoltan
   *
   * This code is only compiled if ALBANY_MPI and ALBANY_ZOLTAN are true
   */

#ifdef ALBANY_ZOLTAN // rebalance needs Zoltan

  if(useSerialMesh){

    // trick to avoid hanging
    bulkData->modification_begin();

    if(commT->getRank() == 0){ // read in the mesh on PE 0


      //stk::io::process_mesh_bulk_data(region, *bulkData);
      mesh_data->populate_bulk_data();
      //bulkData = &mesh_data->bulk_data();

      // Read solution from exodus file.
      if (index >= 0) { // User has specified a time step to restart at
        *out << "Restart Index set, reading solution index : " << index << std::endl;
        mesh_data->read_defined_input_fields(index, &missing);
        m_restartDataTime = region.get_state_time(index);
        m_hasRestartSolution = true;
      }
      else if (res_time >= 0) { // User has specified a time to restart at
        *out << "Restart solution time set, reading solution time : " << res_time << std::endl;
        mesh_data->read_defined_input_fields(res_time, &missing);
        m_restartDataTime = res_time;
        m_hasRestartSolution = true;
      }
      else {
        mesh_data->read_defined_input_fields(-1.0, &missing);
        *out << "Neither restart index or time are set. We still read defined fields in case they are needed (e.g., parameters)."<< std::endl;

//        *out << "Neither restart index or time are set. Not reading solution data from exodus file"<< std::endl;

      }
    }
    else {
      // trick to avoid hanging
      bulkData->modification_begin(); bulkData->modification_begin();
    }

    bulkData->modification_end();

  } // End UseSerialMesh - reading mesh on PE 0

  else
#endif

    /*
     * The following code block reads a single mesh when Albany is compiled serially, or a
     * Nemspread fileset if ALBANY_MPI is true.
     *
     */

  { // running in Serial or Parallel read from Nemspread files
    mesh_data->populate_bulk_data();
    if (!usePamgen)
    {
      // Read solution from exodus file.
      if (index >= 0)
      { // User has specified a time step to restart at
        *out << "Restart Index set, reading solution index : " << index << std::endl;
        mesh_data->read_defined_input_fields(index, &missing);
        m_restartDataTime = region.get_state_time(index);
        m_hasRestartSolution = true;
      }
      else if (res_time >= 0)
      { // User has specified a time to restart at
        *out << "Restart solution time set, reading solution time : " << res_time << std::endl;
        mesh_data->read_defined_input_fields(res_time, &missing);
        m_restartDataTime = res_time;
        m_hasRestartSolution = true;
      }
      else
      {
        *out << "Neither restart index or time are set. We still read defined fields in case they are needed (e.g., parameters)."<< std::endl;
//        *out << "Restart Index not set. Not reading solution from exodus (" << index << ")"<< std::endl;
      }
    }

    bulkData->modification_end();

  } // End Parallel Read - or running in serial

  if(m_hasRestartSolution){

    Teuchos::Array<std::string> default_field;
    default_field.push_back("solution");
    Teuchos::Array<std::string> restart_fields =
      params->get<Teuchos::Array<std::string> >("Restart Fields", default_field);

    // Get the fields to be used for restart

    // See what state data was initialized from the stk::io request
    // This should be propagated into stk::io
    const Ioss::ElementBlockContainer& elem_blocks = region.get_element_blocks();

    /*
    // Uncomment to print what fields are in the exodus file
    Ioss::NameList exo_fld_names;
    elem_blocks[0]->field_describe(&exo_fld_names);
    for(std::size_t i = 0; i < exo_fld_names.size(); i++){
    *out << "Found field \"" << exo_fld_names[i] << "\" in exodus file" << std::endl; } */

    for (std::size_t i=0; i<sis->size(); i++) { Albany::StateStruct& st = *((*sis)[i]);
      if(elem_blocks[0]->field_exists(st.name))

        for(std::size_t j = 0; j < restart_fields.size(); j++)

          if(boost::iequals(st.name, restart_fields[j])){

            *out << "Restarting from field \"" << st.name << "\" found in exodus file." << std::endl;
            st.restartDataAvailable = true;
            break;

          }
    }
  }

#ifdef ALBANY_FELIX
  // Load required fields
  stk::mesh::Selector select_owned_in_part = stk::mesh::Selector(metaData->universal_part()) & stk::mesh::Selector(metaData->locally_owned_part());

  stk::mesh::Selector select_overlap_in_part = stk::mesh::Selector(metaData->universal_part()) & (stk::mesh::Selector(metaData->locally_owned_part()) | stk::mesh::Selector(metaData->globally_shared_part()));

  std::vector<stk::mesh::Entity> nodes;
  stk::mesh::get_selected_entities(select_overlap_in_part, bulkData->buckets(stk::topology::NODE_RANK), nodes);

  std::vector<stk::mesh::Entity> elems;
  stk::mesh::get_selected_entities(select_owned_in_part, bulkData->buckets(stk::topology::ELEM_RANK), elems);

  GO numOwnedNodes(0);
  GO numOwnedElems(0);
  numOwnedNodes = stk::mesh::count_selected_entities(select_owned_in_part, bulkData->buckets(stk::topology::NODE_RANK));
  numOwnedElems = stk::mesh::count_selected_entities(select_owned_in_part, bulkData->buckets(stk::topology::ELEM_RANK));

  GO numGlobalVertices = 0;
  GO numGlobalElements = 0;
  Teuchos::reduceAll<int, GO>(*commT, Teuchos::REDUCE_SUM, 1, &numOwnedNodes, &numGlobalVertices);
  Teuchos::reduceAll<int, GO>(*commT, Teuchos::REDUCE_SUM, 1, &numOwnedElems, &numGlobalElements);

  if (commT->getRank() == 0)
  {
    *out << "Checking if requirements are already stored in the mesh. If not, we import them from ascii files.\n";
  }

  Teuchos::Array<GO> nodeIndices(nodes.size()), elemIndices(elems.size());
  for (int i = 0; i < nodes.size(); ++i)
    nodeIndices[i] = bulkData->identifier(nodes[i]) - 1;
  for (int i = 0; i < elems.size(); ++i)
    elemIndices[i] = bulkData->identifier(elems[i]) - 1;


  // Creating the serial and parallel node maps
  const Tpetra::global_size_t INVALID = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid ();

  Teuchos::RCP<const Tpetra_Map> nodes_map = Tpetra::createNonContigMapWithNode<LO, GO> (nodeIndices, commT, KokkosClassic::Details::getNode<KokkosNode>());
  Teuchos::RCP<const Tpetra_Map> elems_map = Tpetra::createNonContigMapWithNode<LO, GO> (elemIndices, commT, KokkosClassic::Details::getNode<KokkosNode>());

  int numMyNodes = (commT->getRank() == 0) ? numGlobalVertices : 0;
  int numMyElements = (commT->getRank() == 0) ? numGlobalElements : 0;
  Teuchos::RCP<const Tpetra_Map> serial_nodes_map = Teuchos::rcp(new const Tpetra_Map(INVALID, numMyNodes, 0, commT));
  Teuchos::RCP<const Tpetra_Map> serial_elems_map = Teuchos::rcp(new const Tpetra_Map(INVALID, numMyElements, 0, commT));

  // Creating the Tpetra_Import object (to transfer from serial to parallel vectors)
  Tpetra_Import importOperatorNode (serial_nodes_map, nodes_map);
  Tpetra_Import importOperatorElem (serial_elems_map, elems_map);


  Teuchos::ParameterList dummyList;
  Teuchos::ParameterList* req_fields_info;
  if (params->isSublist("Required Fields Info"))
    req_fields_info = &params->sublist("Required Fields Info");
  else
    req_fields_info = &dummyList;

  for (AbstractFieldContainer::FieldContainerRequirements::const_iterator it=req.begin(); it!=req.end(); ++it)
  {
    // Get the file name
    std::string temp_str = *it + " File Name";
    std::string fname = req_fields_info->get<std::string>(temp_str,"");

    // Ge the file type (if not specified, assume Scalar)
    temp_str = *it + " Field Type";
    std::string ftype = req_fields_info->get<std::string>(temp_str,"Node Scalar");

    stk::mesh::Entity node, elem;
    stk::mesh::EntityId nodeId, elemId;
    int lid;
    double* values;

    typedef AbstractSTKFieldContainer::QPScalarFieldType  QPScalarFieldType;
    typedef AbstractSTKFieldContainer::QPVectorFieldType  QPVectorFieldType;
    typedef AbstractSTKFieldContainer::ScalarFieldType    ScalarFieldType;
    typedef AbstractSTKFieldContainer::VectorFieldType    VectorFieldType;

    // Depending on the field type, we need to use different pointers
    if (ftype == "Node Scalar")
    {
      // Creating the serial and (possibly) parallel Tpetra service vectors
      Tpetra_Vector serial_req_vec(serial_nodes_map);
      Tpetra_Vector req_vec(nodes_map);

      temp_str = *it + " Value";
      if (req_fields_info->isParameter(temp_str))
      {
        *out << "Discarding other info about Node Scalar field " << *it << " and filling it with constant value " << req_fields_info->get<double>(temp_str) << "\n";
        // For debug, we allow to fill the field with a given uniform value
        fillTpetraVec (serial_req_vec,req_fields_info->get<double>(temp_str));
      }
      else
      {
        if (fname=="")
        {
          // OK, here's the deal: if the user does not specify the file name or a
          // fixed value for one of the requirements, I assume that that field is already
          // loaded from the mesh. If not, something's wrong and we issue an error
          for (int i(0); i<missing.size(); ++i)
          {
            if (missing[i].field()->name()==*it)
              TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! The Node Scalar field " << *it << " is required. Since it is not present in the mesh file, you must specify the name of an ascii file to load it from.\n");
          }

          // The field is already loaded from the mesh. We skip it.
          *out << "Using mesh-stored values for Node Scalar field " << *it << " since no constant value nor filename has been specified\n";
          continue;
        }

        *out << "Reading Node Scalar field " << *it << " from file " << fname << "\n";
        // Read the input file and stuff it in the Tpetra vector
        readScalarFileSerial (fname,serial_req_vec,commT);
      }

      // Fill the (possibly) parallel vector
      req_vec.doImport(serial_req_vec,importOperatorNode,Tpetra::INSERT);

      // Extracting the mesh field and the tpetra vector view
      ScalarFieldType* field = metaData->get_field<ScalarFieldType>(stk::topology::NODE_RANK, *it);

      TEUCHOS_TEST_FOR_EXCEPTION (field==0, std::logic_error, "Error! Field not present (perhaps is 'Elem Scalar'?).\n");

      Teuchos::ArrayRCP<const ST> req_vec_view = req_vec.get1dView();

      //Now we have to stuff the vector in the mesh data
      for (int i(0); i<nodes.size(); ++i)
      {
        node   = bulkData->get_entity(stk::topology::NODE_RANK, i + 1);
        nodeId = bulkData->identifier(nodes[i]) - 1;
        lid    = nodes_map->getLocalElement((GO)(nodeId));

        values = stk::mesh::field_data(*field, node);
        values[0] = req_vec_view[lid];
      }
    }
    else if (ftype == "Elem Scalar")
    {
      // Creating the serial and (possibly) parallel Tpetra service vectors
      Tpetra_Vector serial_req_vec(serial_elems_map);
      Tpetra_Vector req_vec(elems_map);

      temp_str = *it + " Value";
      if (req_fields_info->isParameter(temp_str))
      {
        *out << "Discarding other info about Elem Scalar field " << *it << " and filling it with constant value " << req_fields_info->get<double>(temp_str) << "\n";
        // For debug, we allow to fill the field with a given uniform value
        fillTpetraVec (serial_req_vec,req_fields_info->get<double>(temp_str));
      }
      else
      {
        if (fname=="")
        {
          // OK, here's the deal: if the user does not specify the file name or a
          // fixed value for one of the requirements, I assume that that field is already
          // loaded from the mesh. If not, something's wrong and we issue an error
          for (int i(0); i<missing.size(); ++i)
          {
            if (missing[i].field()->name()==*it)
              TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! The Elem Scalar field " << *it << " is required. Since it is not present in the mesh file, you must specify the name of an ascii file to load it from.\n");
          }

          // The field is already loaded from the mesh. We skip it.
          *out << "Using mesh-stored values for Elem Scalar field " << *it << " since no constant value nor filename has been specified\n";
          continue;
        }

        *out << "Reading Elem Scalar field " << *it << " from file " << fname << "\n";

        // Read the input file and stuff it in the Tpetra vector
        readScalarFileSerial (fname,serial_req_vec,commT);
      }
      // Fill the (possibly) parallel vector
      req_vec.doImport(serial_req_vec,importOperatorElem,Tpetra::INSERT);

      // Extracting the mesh field and the tpetra vector view
      QPScalarFieldType* field = metaData->get_field<QPScalarFieldType>(stk::topology::ELEM_RANK, *it);
      TEUCHOS_TEST_FOR_EXCEPTION (field==0, std::logic_error, "Error! Field not present (perhaps is 'Node Scalar'?).\n");

      Teuchos::ArrayRCP<const ST> req_vec_view = req_vec.get1dView();

      //Now we have to stuff the vector in the mesh data
      for (int i(0); i<elems.size(); ++i)
      {
        elem   = bulkData->get_entity(stk::topology::ELEM_RANK, i + 1);
        elemId = bulkData->identifier(elems[i]) - 1;
        lid    = elems_map->getLocalElement((GO)(elemId));

        values = stk::mesh::field_data(*field, elem);
        values[0] = req_vec_view[lid];
      }
    }
    else if (ftype == "Node Vector")
    {
      // Loading the dimension of the Vector Field (by default equal to the mesh dimension)
      temp_str = *it + " Field Dimension";
      int fieldDim = req_fields_info->get<int>(temp_str,this->meshSpecs[0]->numDim);

      // Creating the serial and (possibly) parallel Tpetra service multivectors
      Tpetra_MultiVector serial_req_mvec(serial_nodes_map,fieldDim);
      Tpetra_MultiVector req_mvec(nodes_map,fieldDim);

      temp_str = *it + " Value";
      if (req_fields_info->isParameter(temp_str))
      {
        *out << "Discarding other info about Node Vector field " << *it << " and filling it with constant value "
             << req_fields_info->get<Teuchos::Array<double> >(temp_str) << "\n";
        // For debug, we allow to fill the field with a given uniform value
        fillTpetraMVec (serial_req_mvec,req_fields_info->get<Teuchos::Array<double> >(temp_str));
      }
      else
      {
        if (fname=="")
        {
          // OK, here's the deal: if the user does not specify the file name or a
          // fixed value for one of the requirements, I assume that that field is already
          // loaded from the mesh. If not, something's wrong and we issue an error
          for (int i(0); i<missing.size(); ++i)
          {
            if (missing[i].field()->name()==*it)
              TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! The Node Vector field " << *it << " is required. Since it is not present in the mesh file, you must specify the name of an ascii file to load it from.\n");
          }

          // The field is already loaded from the mesh. We skip it.
          *out << "Using mesh-stored values for Node Vector field " << *it << " since no constant value nor filename has been specified\n";
          continue;
        }

        *out << "Reading Node Vector field " << *it << " from file " << fname << "\n";

        // Read the input file and stuff it in the Tpetra multivector
        readVectorFileSerial (fname,serial_req_mvec,commT);
      }

      // Fill the (possibly) parallel vector
      req_mvec.doImport(serial_req_mvec,importOperatorNode,Tpetra::INSERT);

      // Extracting the mesh field and the tpetra vector views
      VectorFieldType* field = metaData->get_field<VectorFieldType>(stk::topology::NODE_RANK, *it);
      TEUCHOS_TEST_FOR_EXCEPTION (field==0, std::logic_error, "Error! Field not present (perhaps is 'Elem Vector'?).\n");

      std::vector<Teuchos::ArrayRCP<const ST> > req_mvec_view;
      for (int i(0); i<fieldDim; ++i)
        req_mvec_view.push_back(req_mvec.getVector(i)->get1dView());

      //Now we have to stuff the vector in the mesh data
      for (int i(0); i<nodes.size(); ++i)
      {
        node   = bulkData->get_entity(stk::topology::NODE_RANK, i + 1);
        nodeId = bulkData->identifier(nodes[i]) - 1;
        lid    = nodes_map->getLocalElement((GO)(nodeId));

        values = stk::mesh::field_data(*field, node);

        for (int iDim(0); iDim<fieldDim; ++iDim)
          values[iDim] = req_mvec_view[iDim][lid];
      }
    }
    else if (ftype == "Elem Vector")
    {
      // Loading the dimension of the Vector Field (by default equal to the mesh dimension)
      temp_str = *it + " Field Dimension";
      int fieldDim = req_fields_info->get<int>(temp_str,this->meshSpecs[0]->numDim);

      // Creating the serial and (possibly) parallel Tpetra service multivectors
      Tpetra_MultiVector serial_req_mvec(serial_elems_map,fieldDim);
      Tpetra_MultiVector req_mvec(elems_map,fieldDim);

      temp_str = *it + " Value";
      if (req_fields_info->isParameter(temp_str))
      {
        *out << "Discarding other info about Elem Vector field " << *it << " and filling it with constant value "
             << req_fields_info->get<Teuchos::Array<double> >(temp_str) << "\n";
        // For debug, we allow to fill the field with a given uniform value
        fillTpetraMVec (serial_req_mvec,req_fields_info->get<Teuchos::Array<double> >(temp_str));
      }
      else
      {
        if (fname=="")
        {
          // OK, here's the deal: if the user does not specify the file name or a
          // fixed value for one of the requirements, I assume that that field is already
          // loaded from the mesh. If not, something's wrong and we issue an error
          for (int i(0); i<missing.size(); ++i)
          {
            if (missing[i].field()->name()==*it)
              TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! The Elem Vector field " << *it << " is required. Since it is not present in the mesh file, you must specify the name of an ascii file to load it from.\n");
          }

          // The field is already loaded from the mesh. We skip it.
          *out << "Using mesh-stored values for Elem Vector field " << *it << " since no constant value nor filename has been specified\n";
          continue;
        }

        *out << "Reading Elem Vector field " << *it << " from file " << fname << "\n";

        // Read the input file and stuff it in the Tpetra multivector
        readVectorFileSerial (fname,serial_req_mvec,commT);
      }
      // Fill the (possibly) parallel vector
      req_mvec.doImport(serial_req_mvec,importOperatorNode,Tpetra::INSERT);

      // Extracting the mesh field and the tpetra vector views
      VectorFieldType* field = metaData->get_field<VectorFieldType>(stk::topology::ELEM_RANK, *it);
      TEUCHOS_TEST_FOR_EXCEPTION (field==0, std::logic_error, "Error! Field not present (perhaps is 'Node Vector'?).\n");
      std::vector<Teuchos::ArrayRCP<const ST> > req_mvec_view;
      for (int i(0); i<fieldDim; ++i)
        req_mvec_view.push_back(req_mvec.getVector(i)->get1dView());

      //Now we have to stuff the vector in the mesh data
      for (int i(0); i<elems.size(); ++i)
      {
        elem   = bulkData->get_entity(stk::topology::ELEM_RANK, i + 1);
        elemId = bulkData->identifier(elems[i]) - 1;
        lid    = elems_map->getLocalElement((GO)(elemId));

        values = stk::mesh::field_data(*field, node);

        for (int iDim(0); iDim<fieldDim; ++iDim)
          values[iDim] = req_mvec_view[iDim][lid];
      }
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameterValue,
                                  "Sorry, I haven't yet implemented the case of field that are not Scalar nor Vector or that is not at nodal nor elemental.\n");
    }
  }

  if (params->get<bool>("Write points coordinates to ascii file", false))
  {
    AbstractSTKFieldContainer::VectorFieldType* coordinates_field = fieldContainer->getCoordinatesField();
    std::ofstream ofile;
    ofile.open("coordinates.ascii");
    if (!ofile.is_open())
    {
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Cannot open coordinates file.\n");
    }

    ofile << nodes.size() << " " << 4 << "\n";

    stk::mesh::Entity node;

    for (int i(0); i<nodes.size(); ++i)
    {
      node   = bulkData->get_entity(stk::topology::NODE_RANK, i + 1);
      double* coord = stk::mesh::field_data(*coordinates_field, node);

      ofile << bulkData->identifier (nodes[i]) << " " << coord[0] << " " << coord[1]  << " " << coord[2] << "\n";
    }

    ofile.close();
  }
#endif

  // Refine the mesh before starting the simulation if indicated
  uniformRefineMesh(commT);

  // Rebalance the mesh before starting the simulation if indicated
  rebalanceInitialMeshT(commT);

  // Build additional mesh connectivity needed for mesh fracture (if indicated)
  computeAddlConnectivity();
}

double
Albany::IossSTKMeshStruct::getSolutionFieldHistoryStamp(int step) const
{
  TEUCHOS_ASSERT(step >= 0 && step < m_solutionFieldHistoryDepth);

  const int index = step + 1; // 1-based step indexing
  const Ioss::Region &  inputRegion = *(mesh_data->get_input_io_region());
  return inputRegion.get_state_time(index);
}

void
Albany::IossSTKMeshStruct::loadSolutionFieldHistory(int step)
{
  TEUCHOS_ASSERT(step >= 0 && step < m_solutionFieldHistoryDepth);

  const int index = step + 1; // 1-based step indexing
  mesh_data->read_defined_input_fields(index);
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::IossSTKMeshStruct::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getValidGenericSTKParameters("Valid IOSS_DiscParams");
  validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic a mesh");
  validPL->set<std::string>("Exodus Input File Name", "", "File Name For Exodus Mesh Input");
  validPL->set<std::string>("Pamgen Input File Name", "", "File Name For Pamgen Mesh Input");
  validPL->set<int>("Restart Index", 1, "Exodus time index to read for inital guess/condition.");
  validPL->set<double>("Restart Time", 1.0, "Exodus solution time to read for inital guess/condition.");
  validPL->set<Teuchos::ParameterList>("Required Fields Info",Teuchos::ParameterList());
  validPL->set<bool>("Write points coordinates to ascii file", "", "Write the mesh points coordinates to file?");

  Teuchos::Array<std::string> emptyStringArray;
  validPL->set<Teuchos::Array<std::string> >("Additional Node Sets", emptyStringArray, "Declare additional node sets not present in the input file");

  return validPL;
}

#ifdef ALBANY_FELIX

void Albany::IossSTKMeshStruct::readScalarFileSerial (std::string& fname,
                                                Tpetra_MultiVector& content,
                                                const Teuchos::RCP<const Teuchos_Comm>& comm) const
{
  GO numNodes;
  Teuchos::ArrayRCP<ST> content_nonConstView = content.get1dViewNonConst();
  if (comm->getRank() == 0)
  {
    std::ifstream ifile;
    ifile.open(fname.c_str());
    if (ifile.is_open())
    {
      ifile >> numNodes;
      TEUCHOS_TEST_FOR_EXCEPTION (numNodes != content.getLocalLength(), Teuchos::Exceptions::InvalidParameterValue,
                                  std::endl << "Error in ExtrudedSTKMeshStruct: Number of nodes in file " << fname << " (" << numNodes << ") is different from the number expected (" << content.getLocalLength() << ")" << std::endl);

      for (GO i = 0; i < numNodes; i++)
        ifile >> content_nonConstView[i];

      ifile.close();
    }
    else
    {
      std::cout << "Warning in IossSTKMeshStruct: unable to open the file " << fname << std::endl;
    }
  }
}

void Albany::IossSTKMeshStruct::readVectorFileSerial (std::string& fname,
                                                      Tpetra_MultiVector& contentVec,
                                                      const Teuchos::RCP<const Teuchos_Comm>& comm) const
{
  GO numNodes;
  int numComponents;
  if (comm->getRank() == 0)
  {
    std::ifstream ifile;
    ifile.open(fname.c_str());
    if (ifile.is_open())
    {
      ifile >> numNodes >> numComponents;
      TEUCHOS_TEST_FOR_EXCEPTION (numNodes != contentVec.getLocalLength(), Teuchos::Exceptions::InvalidParameterValue,
                                  std::endl << "Error in ExtrudedSTKMeshStruct: Number of nodes in file " << fname << " (" << numNodes << ") is different from the number expected (" << contentVec.getLocalLength() << ")" << std::endl);
      TEUCHOS_TEST_FOR_EXCEPTION(numComponents != contentVec.getNumVectors(), Teuchos::Exceptions::InvalidParameterValue,
          std::endl << "Error in ExtrudedSTKMeshStruct: Number of components in file " << fname << " (" << numComponents << ") is different from the number expected (" << contentVec.getNumVectors() << ")" << std::endl);

      for (int icomp(0); icomp<numComponents; ++icomp)
      {
        Teuchos::ArrayRCP<ST> contentVec_nonConstView = contentVec.getVectorNonConst(icomp)->get1dViewNonConst();
        for (GO i = 0; i < numNodes; i++)
          ifile >> contentVec_nonConstView[i];
      }
      ifile.close();
    }
    else
    {
      std::cout << "Warning in IossSTKMeshStruct: unable to open the file " << fname << std::endl;
    }
  }
}

void Albany::IossSTKMeshStruct::fillTpetraVec (Tpetra_Vector& vec, double value)
{
  int numElements = vec.getMap()->getNodeNumElements();
  Teuchos::ArrayRCP<Tpetra_Vector::scalar_type> vec_vals = vec.get1dViewNonConst();
  for (int i(0); i<numElements; ++i)
  {
    vec_vals[i] = value;
  }
}

void Albany::IossSTKMeshStruct::fillTpetraMVec (Tpetra_MultiVector& mvec, const Teuchos::Array<double>& values)
{
  int numElements = mvec.getMap()->getNodeNumElements();
  int numVectors = mvec.getNumVectors();

  TEUCHOS_TEST_FOR_EXCEPTION (numVectors!=values.size(),Teuchos::Exceptions::InvalidParameterValue,
                              "Error! Length of given array of values does not match the field dimension.\n");

  for (int iv(0); iv<numVectors; ++iv)
  {
    Teuchos::ArrayRCP<Tpetra_MultiVector::scalar_type> vec_vals = mvec.getVectorNonConst(iv)->get1dViewNonConst();
    for (int i(0); i<numElements; ++i)
    {
      vec_vals[i] = values[iv];
    }
  }
}

#endif // ALBANY_FELIX

#endif // ALBANY_SEACAS
