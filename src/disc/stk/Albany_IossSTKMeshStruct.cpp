//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_IossSTKMeshStruct.hpp"

#ifdef ALBANY_SEACAS

#include <iostream>

#include "Teuchos_VerboseObject.hpp"

#include <Shards_BasicTopologies.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_io/IossBridge.hpp>
#include <Ioss_SubSystem.h>

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

} // Anonymous namespace

Albany::IossSTKMeshStruct::IossSTKMeshStruct(
                                             const Teuchos::RCP<Teuchos::ParameterList>& params_,
                                             const Teuchos::RCP<const Teuchos_Comm>& commT,
					     const int numParams_) :
  GenericSTKMeshStruct(params_, -1, numParams_),
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  useSerialMesh(false),
  periodic(params->get("Periodic BC", false)),
  m_hasRestartSolution(false),
  m_restartDataTime(-1.0),
  m_solutionFieldHistoryDepth(0)
{
  params->validateParameters(*getValidDiscretizationParameters(),0);

  usePamgen = (params->get("Method","Exodus") == "Pamgen");


  std::vector<std::string> entity_rank_names = stk::mesh::entity_rank_names();

  const Teuchos::MpiComm<int>* theComm = dynamic_cast<const Teuchos::MpiComm<int>* > (commT.get());

  mesh_data = Teuchos::rcp(new stk::io::StkMeshIoBroker(*theComm->getRawMpiComm()));

  // Use Greg Sjaardema's capability to repartition on the fly.
  //    Several partitioning choices: rcb, rib, hsfc, kway, kway-gemo, linear, random
  //          linear does not require Zoltan or metis
  if (params->get<bool>("Use Serial Mesh", false) && commT->getSize() > 1){
  //    Option  external  reads the nemesis files, and must be the default
#ifdef ALBANY_ZOLTAN
    mesh_data->property_add(Ioss::Property("DECOMPOSITION_METHOD", "rib"));
#else
    mesh_data->property_add(Ioss::Property("DECOMPOSITION_METHOD", "linear"));
#endif
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
      if (distrFactorfield != NULL){
        stk::mesh::put_field_on_mesh(*distrFactorfield, newNodeSet, nullptr);
      }
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
        partVec.push_back(part);
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

  // the method 'initializesidesetmeshspecs' requires that ss parts store a valid stk topology.
  // therefore, we try to retrieve the topology of this part using stk stuff.
  auto r = mesh_data->get_input_io_region();
  auto& sss = r->get_sidesets();
  for (auto ss : sss) {
    auto& ssb = ss->get_side_blocks();
    if (ssb.size()==0) { continue; }
    TEUCHOS_TEST_FOR_EXCEPTION(ssb.size()==0, std::runtime_error, "Error! There is a sideset (" + ss->name() + ") in the input mesh with zero side sets.\n");

    const auto* ioss_topo = ssb[0]->topology();
    TEUCHOS_TEST_FOR_EXCEPTION(ioss_topo==nullptr, std::runtime_error, "I give up. No topology in the input mesh for side set " + ss->name() << ".\n");

    auto stk_topo = stk::io::map_ioss_topology_to_stk(ioss_topo, metaData->spatial_dimension());

    auto part = metaData->get_part(ss->name());
    stk::mesh::set_topology(*part,stk_topo);
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

  //Get Cubature Rule
  const std::string cub_rule_string = params->get("Cubature Rule", "GAUSS");
  Intrepid2::EPolyType cub_rule;
  if (cub_rule_string == "GAUSS")
    cub_rule = static_cast<Intrepid2::EPolyType>(Intrepid2::POLYTYPE_GAUSS);
  else if (cub_rule_string == "GAUSS_RADAU_LEFT")
    cub_rule = static_cast<Intrepid2::EPolyType>(Intrepid2::POLYTYPE_GAUSS_RADAU_LEFT);
  else if (cub_rule_string == "GAUSS_RADAU_RIGHT")
    cub_rule = static_cast<Intrepid2::EPolyType>(Intrepid2::POLYTYPE_GAUSS_RADAU_RIGHT);
  else if (cub_rule_string == "GAUSS_LOBATTO")
    cub_rule = static_cast<Intrepid2::EPolyType>(Intrepid2::POLYTYPE_GAUSS_LOBATTO);
  else
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameterValue,
                                "Invalid Cubature Rule: " << cub_rule_string << "; valid options are GAUSS, GAUSS_RADAU_LEFT, GAUSS_RADAU_RIGHT, and GAUSS_LOBATTO");

  int worksetSizeMax = params->get<int>("Workset Size", DEFAULT_WORKSET_SIZE);

  // Get number of elements per element block using Ioss for use
  // in calculating an upper bound on the worksetSize.
  std::vector<int> el_blocks;
  get_element_block_sizes(*mesh_data, el_blocks);
  TEUCHOS_TEST_FOR_EXCEPT(el_blocks.size() != partVec.size());

  int ebSizeMax =  *std::max_element(el_blocks.begin(), el_blocks.end());
  int worksetSize = this->computeWorksetSize(worksetSizeMax, ebSizeMax);

  // Build a map to get the EB name given the index
  for (int eb=0; eb<numEB; eb++) {
    stk::topology stk_topo_data = metaData->get_topology( *partVec[eb] );
    shards::CellTopology shards_ctd = stk::mesh::get_cell_topology(stk_topo_data); 
// Fill in all the various element block lists in Albany_AbstractSTKMeshStruct base class
    this->addElementBlockInfo(eb, partVec[eb]->name(), partVec[eb], shards_ctd);
  }

  // Construct MeshSpecsStruct
  if (!params->get("Separate Evaluators by Element Block",false)) {
    const CellTopologyData& ctd = *elementBlockTopologies_[0].getCellTopologyData();
    this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(
        ctd, numDim, cub, nsNames, ssNames, worksetSize, partVec[0]->name(),
        ebNameToIndex, this->interleavedOrdering, false, cub_rule));
    if (worksetSize == ebSizeMax) this->meshSpecs[0]->singleWorksetSizeAllocation = true;
  } else {
    *out << "MULTIPLE Elem Block in Ioss: DO worksetSize[eb] max?? " << std::endl;
    this->allElementBlocksHaveSamePhysics=false;
    this->meshSpecs.resize(numEB);
    for (int eb=0; eb<numEB; eb++) {
      const CellTopologyData& ctd = *elementBlockTopologies_[eb].getCellTopologyData();
      this->meshSpecs[eb] = Teuchos::rcp(new Albany::MeshSpecsStruct(
          ctd, numDim, cub, nsNames, ssNames, worksetSize, partVec[eb]->name(),
          ebNameToIndex, this->interleavedOrdering, true, cub_rule));
      if (worksetSize == ebSizeMax) this->meshSpecs[eb]->singleWorksetSizeAllocation = true;
    }
  }

  {
    const Ioss::Region& inputRegion = *(mesh_data->get_input_io_region());
    m_solutionFieldHistoryDepth = inputRegion.get_property("state_count").get_int();
  }

  // Upon request, add a nodeset for each sideset
  if (params->get<bool>("Build Node Sets From Side Sets",false))
  {
    this->addNodeSetsFromSideSets ();
  }

  // If requested, mark all parts as io parts
  if (params->get<bool>("Set All Parts IO", false))
    this->setAllPartsIO();

  // Create a mesh specs object for EACH side set
  this->initializeSideSetMeshSpecs(commT);

  // Get upper bound on sideset workset sizes by using Ioss element counts on side blocks
  if (worksetSize == ebSizeMax) {
    if (!params->get("Separate Evaluators by Element Block",false)) {
      for (auto ss : sss) {
        // Get maximum sideset size from Ioss
        auto& ssb = ss->get_side_blocks();
        if (ssb.size()==0) { continue; }
        const std::string ssName = ss->name();
        const auto sidesetSizeMax = ssb[0]->entity_count();

        // Set sideset workset size to maximum
        const auto& sideSetMeshSpecs = this->meshSpecs[0]->sideSetMeshSpecs;
        const auto sideSetMeshSpecIter = sideSetMeshSpecs.find(ssName);
        TEUCHOS_TEST_FOR_EXCEPTION(sideSetMeshSpecIter == sideSetMeshSpecs.end(), std::runtime_error,
            "Cannot find " << ssName << " in sideSetMeshSpecs!\n");
        sideSetMeshSpecIter->second[0]->worksetSize = sidesetSizeMax;
        sideSetMeshSpecIter->second[0]->singleWorksetSizeAllocation = true;
      }
    } else { // FIXME: All element blocks have the same sidesets?
      for (int eb = 0; eb < numEB; ++eb) {
        for (auto ss : sss) {
          // Get maximum sideset size from Ioss
          auto& ssb = ss->get_side_blocks();
          if (ssb.size()==0) { continue; }
          const std::string ssName = ss->name();
          const auto sidesetSizeMax = ssb[0]->entity_count();

          // Set sideset workset size to maximum
          const auto& sideSetMeshSpecs = this->meshSpecs[eb]->sideSetMeshSpecs;
          const auto sideSetMeshSpecIter = sideSetMeshSpecs.find(ssName);
          TEUCHOS_TEST_FOR_EXCEPTION(sideSetMeshSpecIter == sideSetMeshSpecs.end(), std::runtime_error,
              "Cannot find " << ssName << " in sideSetMeshSpecs!\n");
          sideSetMeshSpecIter->second[0]->worksetSize = sidesetSizeMax;
          sideSetMeshSpecIter->second[0]->singleWorksetSizeAllocation = true;
        }
      }
    }
  }

  // Initialize the requested sideset mesh struct in the mesh
  this->initializeSideSetMeshStructs(commT);
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
Albany::IossSTKMeshStruct::setFieldData (
          const Teuchos::RCP<const Teuchos_Comm>& commT,
          const AbstractFieldContainer::FieldContainerRequirements& req,
          const Teuchos::RCP<Albany::StateInfoStruct>& sis,
          const unsigned int worksetSize,
          const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis,
          const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& side_set_req)
{
  this->SetupFieldData(commT, req, sis, worksetSize);

  if(mesh_data->is_bulk_data_null())
    mesh_data->set_bulk_data(*bulkData);

  *out << "IOSS-STK: number of node sets = " << nsPartVec.size() << std::endl;
  *out << "IOSS-STK: number of side sets = " << ssPartVec.size() << std::endl;

  std::vector<stk::io::MeshField> missing;
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

    if(commT->getRank() == 0){ // read in the mesh on PE 0
      mesh_data->populate_bulk_data();

      if (this->numDim!=3)
      {
        // Try to load 3d coordinates (if present in the input file)
        loadOrSetCoordinates3d(index);
      }

      // Read solution from exodus file.
      if (index >= 0) { // User has specified a time step to restart at
        m_restartDataTime = region.get_state_time(index);
        m_hasRestartSolution = true;
      }
      else if (res_time >= 0) { // User has specified a time to restart at
        m_restartDataTime = res_time;
        m_hasRestartSolution = true;
      }
      else {
        *out << "Neither restart index or time are set. Not reading solution data from exodus file"<< std::endl;
      }
    }
    else {
    }


  } // End UseSerialMesh - reading mesh on PE 0

  else
#endif

    /*
     * The following code block reads a single mesh when Albany is compiled serially, or a
     * Nemspread fileset if ALBANY_MPI is true.
     *
     */

  { // running in Serial or Parallel read from Nemspread files
    if (this->numDim!=3)
    {
      // Try to load 3d coordinates (if present in the input file)
      loadOrSetCoordinates3d(index);
    }

    if (!usePamgen)
    {
      // Read solution from exodus file.
      if (index >= 0)
      { // User has specified a time step to restart at
        m_restartDataTime = region.get_state_time(index);
        m_hasRestartSolution = true;
      }
      else if (res_time >= 0)
      { // User has specified a time to restart at
        m_restartDataTime = res_time;
        m_hasRestartSolution = true;
      }
      else
      {
        *out << "Restart Index not set. Not reading solution from exodus (" << index << ")"<< std::endl;
      }
    }

  } // End Parallel Read - or running in serial

  if(m_hasRestartSolution){

    Teuchos::Array<std::string> default_field = {{"solution", "solution_dot", "solution_dotdot"}};
    auto& restart_fields = params->get<Teuchos::Array<std::string> >("Restart Fields", default_field);

    // Get the fields to be used for restart

    // See what state data was initialized from the stk::io request
    // This should be propagated into stk::io
    const Ioss::NodeBlockContainer&    node_blocks = region.get_node_blocks();

    // Uncomment to print what fields are in the exodus file
    // const Ioss::ElementBlockContainer& elem_blocks = region.get_element_blocks();
    // for (const auto& block : elem_blocks) {
    //   Ioss::NameList exo_eb_fld_names;
    //   block->field_describe(&exo_eb_fld_names);
    //   for (const auto& name : exo_eb_fld_names) {
    //     *out << "Found field \"" << name << "\" in elem blocks of exodus file\n";
    //   }
    // }
    // for (const auto& block : node_blocks) {
    //   Ioss::NameList exo_nb_fld_names;
    //   block->field_describe(&exo_nb_fld_names);
    //   for (const auto& name : exo_nb_fld_names) {
    //     *out << "Found field \"" << name << "\" in node blocks of exodus file\n";
    //   }
    // }

    for (const auto& st_ptr : *sis) {
      auto& st = *st_ptr;
      for (const auto& block : node_blocks) {
        if (block->field_exists(st.name)) {
          for (const auto& restart_field : restart_fields) {
            if (st.name==restart_field) {
              *out << "Restarting from field \"" << st.name << "\" found in exodus file.\n";
              st.restartDataAvailable = true;
              break;
            }
          }
        }
      }
    }

    // Read global mesh variables. Should we emit warnings at all?
    for (auto& it : fieldContainer->getMeshVectorStates()) {
      bool found = mesh_data->get_global (it.first, it.second, false); // Last variable is abort_if_not_found. We don't want that.
      if (!found)
        *out << "  *** WARNING *** Mesh vector state '" << it.first << "' was not found in the mesh database.\n";
    }

    for (auto& it : fieldContainer->getMeshScalarIntegerStates()) {
      bool found = mesh_data->get_global (it.first, it.second, false); // Last variable is abort_if_not_found. We don't want that.
      if (!found)
        *out << "  *** WARNING *** Mesh scalar integer state '" << it.first << "' was not found in the mesh database.\n";
    }

    //Read info for layered mehes.
    bool hasLayeredStructure=true;
    std::vector<double> ltr;
    int ordering;
    GO stride;
    boost::any temp_any;

    std::string state_name = "layer_thickness_ratio";
    hasLayeredStructure &= mesh_data->get_global (state_name, ltr, false);
    if(hasLayeredStructure) fieldContainer->getMeshVectorStates()[state_name] = ltr;
    state_name = "ordering";
    hasLayeredStructure &= mesh_data->get_global (state_name, ordering, false);
    if(hasLayeredStructure) fieldContainer->getMeshScalarIntegerStates()[state_name] = ordering;
    state_name = "stride";
    hasLayeredStructure &= mesh_data->get_global (state_name, temp_any, stk::util::ParameterType::INT64, false);
    if(hasLayeredStructure) {
      stride = boost::any_cast<int64_t>(temp_any);
      fieldContainer->getMeshScalarInteger64States()[state_name] = stride;
    }

    if(hasLayeredStructure) {
      Teuchos::ArrayRCP<double> layerThicknessRatio(ltr.size());
      for(decltype(ltr.size()) i=0; i< ltr.size(); ++i) {
        layerThicknessRatio[i] = ltr[i];
      }
      this->layered_mesh_numbering = Teuchos::rcp(new LayeredMeshNumbering<GO>(stride,static_cast<LayeredMeshOrdering>(ordering),layerThicknessRatio));
    }
  }
  else
  {
    // We put all the fields as 'missing'
    const stk::mesh::FieldVector& fields = metaData->get_fields();
    for (decltype(fields.size()) i=0; i<fields.size(); ++i) {
//      TODO, when compiler allows, replace following with this for performance: missing.emplace_back(fields[i],fields[i]->name());
        missing.push_back(stk::io::MeshField(fields[i],fields[i]->name()));
    }
  }

  // If this is a boundary mesh, the side_map/side_node_map may already be present, so we check
  side_maps_present = true;
  bool coherence = true;
  for (const auto& it : missing)
  {
    if (it.field()->name()=="side_to_cell_map" || it.field()->name()=="side_nodes_ids")
    {
      side_maps_present = false;
      coherence = !coherence; // Both fields should be present or absent so coherence should change exactly twice
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION (!coherence, std::runtime_error, "Error! The maps 'side_to_cell_map' and 'side_nodes_ids' should either both be present or both missing, but only one of them was found in the mesh file.\n");

  if (useSerialMesh)
  {
    // Only proc 0 actually read the mesh, and can confirm whether or not the side maps were present
    // Unfortunately, Teuchos does not have a specialization for type bool when it comes to communicators, so we need ints
    int bool_to_int = side_maps_present ? 1 : 0;
    Teuchos::broadcast(*commT,0,1,&bool_to_int);
    side_maps_present = bool_to_int == 1 ? true : false;
  }

  // Loading required input fields from file
  //this->loadRequiredInputFields (req,commT);

  // Rebalance the mesh before starting the simulation if indicated
  rebalanceInitialMeshT(commT);

  // Check that the nodeset created from sidesets contain the right number of nodes
  this->checkNodeSetsFromSideSetsIntegrity ();

  this->setSideSetFieldData(commT, side_set_req, side_set_sis, worksetSize);
}

void
Albany::IossSTKMeshStruct::setBulkData (
          const Teuchos::RCP<const Teuchos_Comm>& commT,
          const AbstractFieldContainer::FieldContainerRequirements& req,
          const Teuchos::RCP<Albany::StateInfoStruct>& sis,
          const unsigned int worksetSize,
          const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis,
          const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& side_set_req)
{
  mesh_data->add_all_mesh_fields_as_input_fields(); // KL: this adds "solution field"
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

      if (this->numDim!=3)
      {
        // Try to load 3d coordinates (if present in the input file)
        loadOrSetCoordinates3d(index);
      }

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
        *out << "Neither restart index or time are set. Not reading solution data from exodus file"<< std::endl;
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
    bulkData->modification_begin();
    mesh_data->populate_bulk_data();
    if (this->numDim!=3)
    {
      // Try to load 3d coordinates (if present in the input file)
      loadOrSetCoordinates3d(index);
    }

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
        *out << "Restart Index not set. Not reading solution from exodus (" << index << ")"<< std::endl;
      }
    }

    bulkData->modification_end();

  } // End Parallel Read - or running in serial

  if(m_hasRestartSolution){

    Teuchos::Array<std::string> default_field = {{"solution", "solution_dot", "solution_dotdot"}};
    auto& restart_fields = params->get<Teuchos::Array<std::string> >("Restart Fields", default_field);

    // Get the fields to be used for restart

    // See what state data was initialized from the stk::io request
    // This should be propagated into stk::io
    const Ioss::NodeBlockContainer&    node_blocks = region.get_node_blocks();

    // Uncomment to print what fields are in the exodus file
    // const Ioss::ElementBlockContainer& elem_blocks = region.get_element_blocks();
    // for (const auto& block : elem_blocks) {
    //   Ioss::NameList exo_eb_fld_names;
    //   block->field_describe(&exo_eb_fld_names);
    //   for (const auto& name : exo_eb_fld_names) {
    //     *out << "Found field \"" << name << "\" in elem blocks of exodus file\n";
    //   }
    // }
    // for (const auto& block : node_blocks) {
    //   Ioss::NameList exo_nb_fld_names;
    //   block->field_describe(&exo_nb_fld_names);
    //   for (const auto& name : exo_nb_fld_names) {
    //     *out << "Found field \"" << name << "\" in node blocks of exodus file\n";
    //   }
    // }

    for (const auto& st_ptr : *sis) {
      auto& st = *st_ptr;
      for (const auto& block : node_blocks) {
        if (block->field_exists(st.name)) {
          for (const auto& restart_field : restart_fields) {
            if (st.name==restart_field) {
              *out << "Restarting from field \"" << st.name << "\" found in exodus file.\n";
              st.restartDataAvailable = true;
              break;
            }
          }
        }
      }
    }

    // Read global mesh variables. Should we emit warnings at all?
    for (auto& it : fieldContainer->getMeshVectorStates()) {
      bool found = mesh_data->get_global (it.first, it.second, false); // Last variable is abort_if_not_found. We don't want that.
      if (!found)
        *out << "  *** WARNING *** Mesh vector state '" << it.first << "' was not found in the mesh database.\n";
    }

    for (auto& it : fieldContainer->getMeshScalarIntegerStates()) {
      bool found = mesh_data->get_global (it.first, it.second, false); // Last variable is abort_if_not_found. We don't want that.
      if (!found)
        *out << "  *** WARNING *** Mesh scalar integer state '" << it.first << "' was not found in the mesh database.\n";
    }

    //Read info for layered mehes.
    bool hasLayeredStructure=true;
    std::vector<double> ltr;
    int ordering;
    GO stride;
    boost::any temp_any;

    std::string state_name = "layer_thickness_ratio";
    hasLayeredStructure &= mesh_data->get_global (state_name, ltr, false);
    if(hasLayeredStructure) fieldContainer->getMeshVectorStates()[state_name] = ltr;
    state_name = "ordering";
    hasLayeredStructure &= mesh_data->get_global (state_name, ordering, false);
    if(hasLayeredStructure) fieldContainer->getMeshScalarIntegerStates()[state_name] = ordering;
    state_name = "stride";
    hasLayeredStructure &= mesh_data->get_global (state_name, temp_any, stk::util::ParameterType::INT64, false);
    if(hasLayeredStructure) {
      stride = boost::any_cast<int64_t>(temp_any);
      fieldContainer->getMeshScalarInteger64States()[state_name] = stride;
    }

    if(hasLayeredStructure) {
      Teuchos::ArrayRCP<double> layerThicknessRatio(ltr.size());
      for(decltype(ltr.size()) i=0; i< ltr.size(); ++i) {
        layerThicknessRatio[i] = ltr[i];
      }
      this->layered_mesh_numbering = Teuchos::rcp(new LayeredMeshNumbering<GO>(stride,static_cast<LayeredMeshOrdering>(ordering),layerThicknessRatio));
    }
  }
  else
  {
    // We put all the fields as 'missing'
    const stk::mesh::FieldVector& fields = metaData->get_fields();
    for (decltype(fields.size()) i=0; i<fields.size(); ++i) {
//      TODO, when compiler allows, replace following with this for performance: missing.emplace_back(fields[i],fields[i]->name());
        missing.push_back(stk::io::MeshField(fields[i],fields[i]->name()));
    }
  }

  // If this is a boundary mesh, the side_map/side_node_map may already be present, so we check
  side_maps_present = true;
  bool coherence = true;
  for (const auto& it : missing)
  {
    if (it.field()->name()=="side_to_cell_map" || it.field()->name()=="side_nodes_ids")
    {
      side_maps_present = false;
      coherence = !coherence; // Both fields should be present or absent so coherence should change exactly twice
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION (!coherence, std::runtime_error, "Error! The maps 'side_to_cell_map' and 'side_nodes_ids' should either both be present or both missing, but only one of them was found in the mesh file.\n");

  if (useSerialMesh)
  {
    // Only proc 0 actually read the mesh, and can confirm whether or not the side maps were present
    // Unfortunately, Teuchos does not have a specialization for type bool when it comes to communicators, so we need ints
    int bool_to_int = side_maps_present ? 1 : 0;
    Teuchos::broadcast(*commT,0,1,&bool_to_int);
    side_maps_present = bool_to_int == 1 ? true : false;
  }

  // Loading required input fields from file
  this->loadRequiredInputFields (req,commT);

  // Rebalance the mesh before starting the simulation if indicated
  rebalanceInitialMeshT(commT);

  // Check that the nodeset created from sidesets contain the right number of nodes
  this->checkNodeSetsFromSideSetsIntegrity ();

  // Finally, perform the setup of the (possible) side set meshes (including extraction if of type SideSetSTKMeshStruct)
  this->setSideSetBulkData(commT, side_set_req, side_set_sis, worksetSize);

  fieldAndBulkDataSet = true;
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
Albany::IossSTKMeshStruct::loadOrSetCoordinates3d(int index)
{
  const std::string coords3d_name = "coordinates3d";

  Teuchos::RCP<Ioss::Region> region = mesh_data->get_input_io_region();
  const Ioss::NodeBlockContainer& node_blocks = region->get_node_blocks();
  Ioss::NodeBlock *nb = node_blocks[0];

  if (nb->field_exists(coords3d_name) && index>0)
  {
    // The field "coordinates3d" exists in the input mesh
    // (which must then come from a previous Albany run), so load it.
    std::vector<stk::mesh::Entity> nodes;
    stk::mesh::get_entities(*bulkData,stk::topology::NODE_RANK,nodes);

    // This is some trickery to get around STK-Ioss implementation.
    // Basically, you cannot access transient fields if begin_state
    // has not yet been called (with the proper step number).
    // Therefore, if the current state has a step number invalid
    // or different to the one desired, we set it to the restart
    // index state. If the state was already set, we also take
    // care of resetting the index back to the original configuration.
    const int current_step = region->get_current_state();
    if (current_step != index) {
      if (current_step != -1) {
        region->end_state(current_step);
      }
      region->begin_state(index);
    }

    stk::io::field_data_from_ioss(*bulkData, this->getCoordinatesField3d(), nodes, nb, coords3d_name);
    if (current_step != -1) {
      region->begin_state(index);
    }
  }
  else
  {
    if (nb->field_exists(coords3d_name)) {
      // The 3d coords field exists in the input mesh, but the restart index was
      // not set in the input file. We issue a warning, and load default coordinates
      *out << "WARNING! The field 'coordinates3d' was found in the input mesh, but no restart index was specified.\n"
           << "         Albany will set the 3d coordinates to the 'default' ones (filling native coordinates with trailins zeros).\n";
    }
    // The input mesh does not store the 'coordinates3d' field
    // (perhaps the mesh does not come from a previous Albany run).
    // Hence, we initialize coordinates3d with coordinates,
    // and we fill 'extra' dimensions with 0's. Hopefully, this is ok.

    // Use GenericSTKMeshStruct functionality
    this->setDefaultCoordinates3d();
  }
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

#endif // ALBANY_SEACAS
