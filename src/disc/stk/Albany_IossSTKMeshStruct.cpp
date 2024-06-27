//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_IossSTKMeshStruct.hpp"

#ifdef ALBANY_SEACAS

#include "Albany_Utils.hpp"
#include "Albany_CommUtils.hpp"

#include <Teuchos_RCPStdSharedPtrConversions.hpp>
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

#include <iostream>


namespace {

void get_element_block_sizes(stk::io::StkMeshIoBroker &mesh_data,
                             std::vector<int>& el_blocks)
{
  Ioss::Region &io = *mesh_data.get_input_ioss_region();
  const Ioss::ElementBlockContainer& elem_blocks = io.get_element_blocks();
  for(Ioss::ElementBlockContainer::const_iterator it = elem_blocks.begin(); it != elem_blocks.end(); ++it) {
    Ioss::ElementBlock *entity = *it;
    if (stk::io::include_entity(entity)) {
      el_blocks.push_back(entity->get_property("entity_count").get_int());
    }
  }
}

} // Anonymous namespace

namespace Albany
{

IossSTKMeshStruct::
IossSTKMeshStruct(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                  const Teuchos::RCP<const Teuchos_Comm>& comm,
					        const int numParams_)
 : GenericSTKMeshStruct(params_, -1, numParams_)
 , out(Teuchos::VerboseObjectBase::getDefaultOStream())
 , useSerialMesh(false)
 , periodic(params->get("Periodic BC", false))
{
  params->validateParameters(*getValidDiscretizationParameters(),0);

  usePamgen = (params->get("Method","Exodus") == "Pamgen");

  std::vector<std::string> entity_rank_names = stk::mesh::entity_rank_names();

  auto mpiComm = getMpiCommFromTeuchosComm(comm);

  mesh_data = Teuchos::rcp(new stk::io::StkMeshIoBroker(mpiComm));
  mesh_data->use_simple_fields();

  // Use Greg Sjaardema's capability to repartition on the fly.
  //    Several partitioning choices: rcb, rib, hsfc, kway, kway-gemo, linear, random
  //          linear does not require Zoltan or metis
  if (params->get<bool>("Use Serial Mesh", false) && comm->getSize() > 1){
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

  metaData = Teuchos::rcp(mesh_data->meta_data_ptr());

  // End of creating input mesh

  typedef Teuchos::Array<std::string> StringArray;
  const StringArray additionalNodeSets = params->get("Additional Node Sets", StringArray());
  for (const auto& nsn : additionalNodeSets) {
    stk::mesh::Part &newNodeSet = metaData->declare_part(nsn, stk::topology::NODE_RANK);
    if (!stk::io::is_part_io_part(newNodeSet)) {
      stk::mesh::Field<double> * const distrFactorfield = metaData->get_field<double>(stk::topology::NODE_RANK, "distribution_factors");
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

  for (const auto& part : all_parts) {
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

  *out << "IOSS-STK: number of node sets = " << nsPartVec.size() << std::endl;
  *out << "IOSS-STK: number of side sets = " << ssPartVec.size() << std::endl;

  // the method 'initializesidesetmeshspecs' requires that ss parts store a valid stk topology.
  // therefore, we try to retrieve the topology of this part using stk stuff.
  auto r = mesh_data->get_input_ioss_region();
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

  int worksetSizeMax = params->get<int>("Workset Size", -1);

  // Get number of elements per element block using Ioss for use
  // in calculating an upper bound on the worksetSize.
  std::vector<int> el_blocks;
  get_element_block_sizes(*mesh_data, el_blocks);
  TEUCHOS_TEST_FOR_EXCEPT(el_blocks.size() != partVec.size());

  int ebSizeMax =  *std::max_element(el_blocks.begin(), el_blocks.end());
  int worksetSize = computeWorksetSize(worksetSizeMax, ebSizeMax);

  // Build a map to get the EB name given the index
  for (int eb=0; eb<numEB; eb++) {
    stk::topology stk_topo_data = metaData->get_topology( *partVec[eb] );
    shards::CellTopology shards_ctd = stk::mesh::get_cell_topology(stk_topo_data); 
// Fill in all the various element block lists in Albany_AbstractSTKMeshStruct base class
    this->addElementBlockInfo(eb, partVec[eb]->name(), partVec[eb], shards_ctd);
  }

  // Construct MeshSpecsStruct
  const CellTopologyData& ctd = *elementBlockTopologies_[0].getCellTopologyData();
  this->meshSpecs[0] = Teuchos::rcp(new MeshSpecsStruct(
      ctd, numDim, nsNames, ssNames, worksetSize, partVec[0]->name(),
      ebNameToIndex));

  const Ioss::Region& inputRegion = *(mesh_data->get_input_ioss_region());
  m_solutionFieldHistoryDepth = inputRegion.get_property("state_count").get_int();

  // Upon request, add a nodeset for each sideset
  if (params->get<bool>("Build Node Sets From Side Sets",false))
  {
    this->addNodeSetsFromSideSets ();
  }

  // If requested, mark all parts as io parts
  if (params->get<bool>("Set All Parts IO", false))
    this->setAllPartsIO();

  // Create a mesh specs object for EACH side set
  this->initializeSideSetMeshSpecs(comm);

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
        }
      }
    }
  }
}

IossSTKMeshStruct::~IossSTKMeshStruct()
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

void IossSTKMeshStruct::
setFieldData (const Teuchos::RCP<const Teuchos_Comm>& comm,
              const Teuchos::RCP<StateInfoStruct>& sis)
{
  GenericSTKMeshStruct::setFieldData(comm, sis);

  if(mesh_data->is_bulk_data_null())
    mesh_data->set_bulk_data(*bulkData);

  // Restart index to read solution from exodus file.
  if (params->isParameter("Restart Index")) {
    TEUCHOS_TEST_FOR_EXCEPTION (
      params->isParameter("Restart Time"), std::logic_error,
      "Error! Do not provide both 'Restart Index' and 'Restart Time'.\n");

    // User has specified a time step to restart at
    int index = params->get<int>("Restart Index"); // Default to no restart

    const auto& region = *mesh_data->get_input_ioss_region();
    m_restartDataTime = region.get_state_time(index);
    m_hasRestartSolution = true;

    *out << "Restart Index set, reading solution index : " << index << std::endl;
  } else if (params->isParameter("Restart Time")) {
    m_restartDataTime = params->get<double>("Restart Time");
    // User has specified a time to restart at
    m_hasRestartSolution = true;
    *out << "Restart solution time set, reading solution time : " << m_restartDataTime << std::endl;
  } else {
    *out << "Neither restart index or time are set. Not reading solution data from exodus file"<< std::endl;
  }

  if (m_hasRestartSolution) {
    mesh_data->add_all_mesh_fields_as_input_fields(); // KL: this adds "solution field"
    Teuchos::Array<std::string> default_field = {{"solution", "solution_dot", "solution_dotdot"}};
    const auto& restart_fields = params->get<Teuchos::Array<std::string> >("Restart Fields", default_field);

    // Check if states are available in the mesh
    const auto& region = *mesh_data->get_input_ioss_region();
    const auto& node_blocks = region.get_node_blocks();

    for (const auto& st_ptr : *sis) {
      auto& st = *st_ptr;
      if (std::find(restart_fields.begin(),restart_fields.end(),st.name)==restart_fields.end()) {
        continue;
      }
      st.restartDataAvailable = node_blocks.size()>0;
      for (const auto& block : node_blocks) {
        st.restartDataAvailable &= block->field_exists(st.name);
      }
    }
  }

}

void IossSTKMeshStruct::
setBulkData (const Teuchos::RCP<const Teuchos_Comm>& comm)
{
  metaData->commit();

#ifdef ALBANY_ZOLTAN // rebalance needs Zoltan
  // The following code block reads a single mesh on PE 0, then distributes the mesh across
  // the other processors. stk_rebalance is used, which requires Zoltan
  if (useSerialMesh){

    // trick to avoid hanging
    bulkData->modification_begin();

    if(comm->getRank() == 0){
      mesh_data->populate_bulk_data();
    } else {
      // trick to avoid hanging
      bulkData->modification_begin(); bulkData->modification_begin();
    }
    bulkData->modification_end();

  } else
#endif
  {
    // The following code block reads a single mesh when Albany is compiled serially, or a
    // Nemspread fileset if ALBANY_MPI is true.
    bulkData->modification_begin();
    mesh_data->populate_bulk_data();
    bulkData->modification_end();
  }

  // Note: we cannot load fields/coords during setFieldData, since we need a valid bulkData
  std::vector<stk::io::MeshField> missing;
  if (m_hasRestartSolution) {
    mesh_data->read_defined_input_fields(m_restartDataTime, &missing);
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

    for (auto& it : fieldContainer->getMeshScalarInteger64States()) {
      stk::util::Parameter temp_any;
      temp_any.type = stk::util::ParameterType::INT64;
      bool found = mesh_data->get_global (it.first, temp_any, false); // Last variable is abort_if_not_found. We don't want that.
      if (!found)
        *out << "  *** WARNING *** Mesh scalar integer state '" << it.first << "' was not found in the mesh database.\n";
    }
  }

  // If this is a boundary mesh, the side_map/side_node_map may already be present, so we check
  auto region = mesh_data->get_input_ioss_region();
  const auto& elem_blocks = region->get_element_blocks();
  auto *eb = elem_blocks[0];
  side_maps_present = eb->field_exists("side_to_cell_map") and eb->field_exists("side_nodes_ids");
  const bool coherence = eb->field_exists("side_to_cell_map") == eb->field_exists("side_nodes_ids");
  TEUCHOS_TEST_FOR_EXCEPTION (!coherence, std::runtime_error, "Error! The maps 'side_to_cell_map' and 'side_nodes_ids' should either both be present or both missing, but only one of them was found in the mesh file.\n");

  if (useSerialMesh)
  {
    // Only proc 0 actually read the mesh, and can confirm whether or not the side maps were present
    // Unfortunately, Teuchos does not have a specialization for type bool when it comes to communicators, so we need ints
    int bool_to_int = side_maps_present ? 1 : 0;
    Teuchos::broadcast(*comm,0,1,&bool_to_int);
    side_maps_present = bool_to_int == 1 ? true : false;
  }

  // Check if the input mesh is layered (i.e., if it stores layers info)
  std::string state_name = "layer_thickness_ratio";
  if (mesh_data->has_input_global(state_name)) {
    // layer ratios
    std::vector<double> ltr;
    mesh_data->get_global (state_name, ltr, true);
    fieldContainer->getMeshVectorStates()[state_name] = ltr;
    this->mesh_layers_ratio.assign(ltr.begin(),ltr.end());

    // Ordering
    int orderingAsInt;
    state_name = "ordering";
    mesh_data->get_global (state_name, orderingAsInt, true);
    TEUCHOS_TEST_FOR_EXCEPTION (orderingAsInt!=0 && orderingAsInt!=1, std::runtime_error,
        "Error! Invalid ordering (" << orderingAsInt << "). Valid values:\n"
        "   0: LAYER ordering, 1: COLUMN ordering\n");
    fieldContainer->getMeshScalarIntegerStates()[state_name] = orderingAsInt;

    // number of layers
    int numLayers;
    state_name = "num_layers";
    mesh_data->get_global (state_name, numLayers, true);
    TEUCHOS_TEST_FOR_EXCEPTION (numLayers<=0, std::runtime_error,
        "Error! Non-positive number of layers in input mesh (" << numLayers << "\n");
    fieldContainer->getMeshScalarIntegerStates()[state_name] = numLayers;

    // Max global elem id
    state_name = "max_2d_elem_gid";
    stk::util::Parameter temp_any;
    temp_any.type = stk::util::ParameterType::INT64;
    mesh_data->get_global (state_name, temp_any, true);
    auto max_2d_elem_gid = temp_any.get_value<int64_t>();
    fieldContainer->getMeshScalarInteger64States()[state_name] = max_2d_elem_gid ;

    // Max global node id
    state_name = "max_2d_node_gid";
    mesh_data->get_global (state_name, temp_any, true);
    auto max_2d_node_gid = temp_any.get_value<int64_t>();
    fieldContainer->getMeshScalarInteger64States()[state_name] = max_2d_node_gid;

    // Build GO cell layers data
    auto ordering = orderingAsInt==0 ? LayeredMeshOrdering::LAYER : LayeredMeshOrdering::COLUMN;
    this->global_cell_layers_data = Teuchos::rcp(new LayeredMeshNumbering<GO>(max_2d_elem_gid,numLayers,ordering));

    // Build LO cell layers data
    stk::mesh::Selector select_owned_in_part(metaData->locally_owned_part());
    const auto& buckets = bulkData->buckets(stk::topology::ELEM_RANK);
    const LO numLocalCells3D = stk::mesh::count_selected_entities(select_owned_in_part,buckets);
    const LO numLocalCells2D = numLocalCells3D / numLayers;
    this->local_cell_layers_data = Teuchos::rcp(new LayeredMeshNumbering<LO>(numLocalCells2D,numLayers,ordering));

    // Shards has both Hexa and Wedge with bot and top in the last two side positions
    this->global_cell_layers_data->top_side_pos = this->meshSpecs[0]->ctd.side_count - 1;
    this->global_cell_layers_data->bot_side_pos = this->meshSpecs[0]->ctd.side_count - 2;
    this->local_cell_layers_data->top_side_pos = this->meshSpecs[0]->ctd.side_count - 1;
    this->local_cell_layers_data->bot_side_pos = this->meshSpecs[0]->ctd.side_count - 2;
  }

  // Set 3d coords *before* loading fields, or else you won't have x/y/z available
  loadOrSetCoordinates3d();

  // Load any required field from file
  this->loadRequiredInputFields (comm);

  // Rebalance the mesh before starting the simulation if indicated
  rebalanceInitialMesh(comm);

  // Check that the nodeset created from sidesets contain the right number of nodes
  this->checkNodeSetsFromSideSetsIntegrity ();

  m_bulk_data_set = true;
}

void
IossSTKMeshStruct::loadOrSetCoordinates3d()
{
  const std::string coords3d_name = "coordinates3d";

  auto region = mesh_data->get_input_ioss_region();
  const Ioss::NodeBlockContainer& node_blocks = region->get_node_blocks();
  Ioss::NodeBlock *nb = node_blocks[0];

  const int current_step = region->get_current_state();
  const int index = params->isParameter("Restart Index")
                  ? params->get<int>("Restart Index")
                  : current_step;

  if (nb->field_exists(coords3d_name) and index!=-1) {
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

    if (current_step != -1) {
      region->end_state(current_step);
    }

    region->begin_state(index);
    stk::io::field_data_from_ioss(*bulkData, this->getCoordinatesField3d(), nodes, nb, coords3d_name);
    region->end_state(index);

    if (current_step != -1) {
      region->begin_state(current_step);
    }
  } else {
    if (nb->field_exists(coords3d_name)) {
      // The 3d coords field exists in the input mesh, but the restart index was
      // not set in the input file. We issue a warning, and load default coordinates
      *out << "WARNING! The field 'coordinates3d' was found in the input mesh, but no restart index was specified.\n"
           << "         Albany will set the 3d coordinates to the 'default' ones (filling native coordinates with trailing zeros).\n";
    }
    setDefaultCoordinates3d();
  }
}

double
IossSTKMeshStruct::getSolutionFieldHistoryStamp(int step) const
{
  TEUCHOS_ASSERT(step >= 0 && step < m_solutionFieldHistoryDepth);

  const int index = step + 1; // 1-based step indexing
  const Ioss::Region &  inputRegion = *(mesh_data->get_input_ioss_region());
  return inputRegion.get_state_time(index);
}

void
IossSTKMeshStruct::loadSolutionFieldHistory(int step)
{
  TEUCHOS_ASSERT(step >= 0 && step < m_solutionFieldHistoryDepth);

  const int index = step + 1; // 1-based step indexing
  mesh_data->read_defined_input_fields(index);
}

Teuchos::RCP<const Teuchos::ParameterList>
IossSTKMeshStruct::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getValidGenericSTKParameters("Valid IOSS_DiscParams");
  validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic a mesh");
  validPL->set<std::string>("Exodus Input File Name", "", "File Name For Exodus Mesh Input");
  validPL->set<std::string>("Pamgen Input File Name", "", "File Name For Pamgen Mesh Input");
  validPL->set<int>("Restart Index", 1, "Exodus time index to read for initial guess/condition.");
  validPL->set<double>("Restart Time", 1.0, "Exodus solution time to read for initial guess/condition.");
  validPL->set<Teuchos::ParameterList>("Required Fields Info",Teuchos::ParameterList());
  validPL->set<bool>("Write points coordinates to ascii file", "", "Write the mesh points coordinates to file?");

  Teuchos::Array<std::string> emptyStringArray;
  validPL->set<Teuchos::Array<std::string> >("Additional Node Sets", emptyStringArray, "Declare additional node sets not present in the input file");

  return validPL;
}

} // namespace Albany

#endif // ALBANY_SEACAS
