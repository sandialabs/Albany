//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_DiscretizationFactory.hpp"
#include "Albany_GenericSTKMeshStruct.hpp"
#include <Albany_STKNodeSharing.hpp>
#include <Albany_ThyraUtils.hpp>
#include <Albany_CombineAndScatterManager.hpp>
#include <Albany_GlobalLocalIndexer.hpp>

#include "Albany_KokkosTypes.hpp"
#include "Albany_Gather.hpp"
#include "Albany_CommUtils.hpp"
#include "Albany_Utils.hpp"

#include "Albany_OrdinarySTKFieldContainer.hpp"
#include "Albany_MultiSTKFieldContainer.hpp"

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/CreateAdjacentEntities.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>

// Expression reading
#ifdef ALBANY_PANZER_EXPR_EVAL
#include <Panzer_ExprEval_impl.hpp>
#endif

// Rebalance
#ifdef ALBANY_ZOLTAN
#include <percept/stk_rebalance/Rebalance.hpp>
#include <percept/stk_rebalance/Partition.hpp>
#include <percept/stk_rebalance/ZoltanPartition.hpp>
#include <percept/stk_rebalance/RebalanceUtils.hpp>
#endif

#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_RCPStdSharedPtrConversions.hpp"

#include <iostream>

namespace Albany
{

GenericSTKMeshStruct::GenericSTKMeshStruct(
    const Teuchos::RCP<Teuchos::ParameterList>& params_,
    const int numDim_,
    const int numParams_)
    : params(params_),
      num_params(numParams_)
{
  metaData = Teuchos::rcp(new stk::mesh::MetaData());
  metaData->use_simple_fields();

  // numDim = -1 is default flag value to postpone initialization
  if (numDim_>0) {
    this->numDim = numDim_;
    std::vector<std::string> entity_rank_names = stk::mesh::entity_rank_names();
    metaData->initialize(numDim_, entity_rank_names);
  }

  allElementBlocksHaveSamePhysics = true;
  num_time_deriv = params->get<int>("Number Of Time Derivatives");

  requiresAutomaticAura = params->get<bool>("Use Automatic Aura", false);

  // This is typical, can be resized for multiple material problems
  meshSpecs.resize(1);
}

void GenericSTKMeshStruct::
setFieldData (const Teuchos::RCP<const Teuchos_Comm>& comm)
{
  TEUCHOS_TEST_FOR_EXCEPTION(!metaData->is_initialized(), std::logic_error,
       "[GenericSTKMeshStruct::SetupFieldData] metaData->initialize(numDim) not yet called" << std::endl);

  if (bulkData.is_null()) {
     auto mpiComm = getMpiCommFromTeuchosComm(comm);
     stk::mesh::MeshBuilder meshBuilder = stk::mesh::MeshBuilder(mpiComm);
     if(requiresAutomaticAura)
       meshBuilder.set_aura_option(stk::mesh::BulkData::AUTO_AURA);
     else
       meshBuilder.set_aura_option(stk::mesh::BulkData::NO_AUTO_AURA);

     meshBuilder.set_bucket_capacity(meshSpecs[0]->worksetSize);
     meshBuilder.set_add_fmwk_data(false);
     std::unique_ptr<stk::mesh::BulkData> bulkDataPtr = meshBuilder.create(Teuchos::get_shared_ptr(metaData));
     bulkData = Teuchos::rcp(bulkDataPtr.release());
  }

  // Build the container for the STK fields
  Teuchos::Array<std::string> default_solution_vector; // Empty
  Teuchos::Array<Teuchos::Array<std::string> > solution_vector;
  solution_vector.resize(num_time_deriv + 1);
  bool user_specified_solution_components = false;
  solution_vector[0] =
    params->get<Teuchos::Array<std::string> >("Solution Vector Components", default_solution_vector);

  if(solution_vector[0].length() > 0)
     user_specified_solution_components = true;

  if(num_time_deriv >= 1){
    solution_vector[1] =
      params->get<Teuchos::Array<std::string> >("SolutionDot Vector Components", default_solution_vector);
    if(solution_vector[1].length() > 0)
       user_specified_solution_components = true;
  }

  if(num_time_deriv >= 2){
    solution_vector[2] =
      params->get<Teuchos::Array<std::string> >("SolutionDotDot Vector Components", default_solution_vector);
    if(solution_vector[2].length() > 0)
       user_specified_solution_components = true;
  }

  Teuchos::Array<std::string> default_residual_vector; // Empty
  Teuchos::Array<std::string> residual_vector =
    params->get<Teuchos::Array<std::string> >("Residual Vector Components", default_residual_vector);

  // Build the usual Albany fields unless the user explicitly specifies the residual or solution vector layout
  if(user_specified_solution_components && (residual_vector.length() > 0)){
    this->fieldContainer = Teuchos::rcp(new MultiSTKFieldContainer(params,
        metaData, bulkData, numDim, num_params));
  } else {
    this->fieldContainer = Teuchos::rcp(new OrdinarySTKFieldContainer(params,
        metaData, bulkData, numDim, num_params));
  }

// Exodus is only for 2D and 3D. Have 1D version as well
  exoOutput = params->isType<std::string>("Exodus Output File Name");
  if (exoOutput)
    exoOutFile = params->get<std::string>("Exodus Output File Name");
  exoOutputInterval = params->get<int>("Exodus Write Interval", 1);

  //get the type of transformation of STK mesh
  transformType = params->get("Transform Type", "None"); //get the type of transformation of STK mesh
  felixAlpha = params->get("LandIce alpha", 0.0); //for LandIce problems
  felixL = params->get("LandIce L", 1.0); //for LandIce problems
  xShift = params->get("x-shift", 0.0);
  yShift = params->get("y-shift", 0.0);
  zShift = params->get("z-shift", 0.0);
  betas_BLtransform = params->get<Teuchos::Array<double> >("Betas BL Transform",  Teuchos::tuple<double>(0.0, 0.0, 0.0));

  points_per_edge = params->get("Element Degree", 1) + 1;

  //boolean specifying if ascii mesh has contiguous IDs; only used for ascii meshes on 1 processor
  contigIDs = params->get("Contiguous IDs", true);

  //Does user want to write coordinates to matrix market file (e.g., for ML analysis)?
  writeCoordsToMMFile = params->get("Write Coordinates to MatrixMarket", false);

  transferSolutionToCoords = params->get<bool>("Transfer Solution to Coordinates", false);
}

void GenericSTKMeshStruct::setAllPartsIO()
{
#ifdef ALBANY_SEACAS
  for (auto& it : partVec)
  {
    stk::mesh::Part& part = *it;
    if (!stk::io::is_part_io_part(part))
      stk::io::put_io_part_attribute(part);
  }
  for (auto& it : nsPartVec)
  {
    stk::mesh::Part& part = *it.second;
    if (!stk::io::is_part_io_part(part))
      stk::io::put_io_part_attribute(part);
  }
  for (auto& it : ssPartVec)
  {
    stk::mesh::Part& part = *it.second;
    if (!stk::io::is_part_io_part(part))
      stk::io::put_io_part_attribute(part);
  }
#endif
}

void
GenericSTKMeshStruct::cullSubsetParts(std::vector<std::string>& ssNames,
    std::map<std::string, stk::mesh::Part*>& partMap){

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

  for(it = partMap.begin(); it != partMap.end(); ++it){ // loop over the parts in the map

    // for each part in turn, get the name of parts that are a subset of it

    const stk::mesh::PartVector & subsets   = it->second->subsets();

    for ( p = subsets.begin() ; p != subsets.end() ; ++p ) {
      const std::string & n = (*p)->name();
//      std::cout << "Erasing: " << n << std::endl;
      partMap.erase(n); // erase it if it is in the base map
    }
  }

//  ssNames.clear();

  // Build the remaining data structures
  for(it = partMap.begin(); it != partMap.end(); ++it){ // loop over the parts in the map

    std::string ssn = it->first;
    ssNames.push_back(ssn);

  }
}

void GenericSTKMeshStruct::setDefaultCoordinates3d ()
{
  // If the mesh is already a 3d mesh, coordinates_field==coordinates_field3d
  if (this->numDim==3) return;

  // We make coordinates_field3d store the same coordinates as coordinates_field,
  // padding the vector of coordinates with zeros

  std::vector<stk::mesh::Entity> nodes;
  stk::mesh::get_entities(*bulkData,stk::topology::NODE_RANK,nodes);
  double* values;
  double* values3d;
  for (auto node : nodes)
  {
    values3d = stk::mesh::field_data(*this->getCoordinatesField3d(), node);
    values   = stk::mesh::field_data(*this->getCoordinatesField(), node);

    for (int iDim=0; iDim<numDim; ++iDim) {
      values3d[iDim] = values[iDim];
    }
    for (int iDim=numDim; iDim<3; ++iDim) {
      values3d[iDim] = 0.0;
    }
  }
}

void GenericSTKMeshStruct::rebalanceInitialMesh (const Teuchos::RCP<const Teuchos_Comm>& comm){

  bool rebalance = params->get<bool>("Rebalance Mesh", false);

  if(rebalance) {
    TEUCHOS_TEST_FOR_EXCEPTION (this->side_maps_present, std::runtime_error,
                                "Error! Rebalance is not supported when side maps are present.\n");
    rebalanceAdaptedMesh (params, comm);
  }
}

void GenericSTKMeshStruct::
rebalanceAdaptedMesh (const Teuchos::RCP<Teuchos::ParameterList>& params_,
                      const Teuchos::RCP<const Teuchos_Comm>& comm)
{
// Zoltan is required here
#ifdef ALBANY_ZOLTAN

    using std::cout; using std::endl;

    if(comm->getSize() <= 1)

      return;

    double imbalance;

    AbstractSTKFieldContainer::STKFieldType* coordinates_field = fieldContainer->getCoordinatesField();

    stk::mesh::Selector selector(metaData->universal_part());
    stk::mesh::Selector owned_selector(metaData->locally_owned_part());


    if(comm->getRank() == 0){

      std::cout << "Before rebal nelements " << comm->getRank() << "  " <<
        stk::mesh::count_selected_entities(owned_selector, bulkData->buckets(stk::topology::ELEMENT_RANK)) << endl;

      std::cout << "Before rebal " << comm->getRank() << "  " <<
        stk::mesh::count_selected_entities(owned_selector, bulkData->buckets(stk::topology::NODE_RANK)) << endl;
    }


    imbalance = stk::rebalance::check_balance(*bulkData, NULL, stk::topology::NODE_RANK, &selector);

    if(comm->getRank() == 0)

      std::cout << "Before rebalance: Imbalance threshold is = " << imbalance << endl;

    // Use Zoltan to determine new partition. Set the desired parameters (if any) from the input file

    Teuchos::ParameterList graph_options;

   //graph_options.sublist(stk::rebalance::Zoltan::default_parameters_name()).set("LOAD BALANCING METHOD"      , "4");
    //graph_options.sublist(stk::rebalance::Zoltan::default_parameters_name()).set("ZOLTAN DEBUG LEVEL"      , "10");

    if(params_->isSublist("Rebalance Options")){

      const Teuchos::RCP<Teuchos::ParameterList>& load_balance_method = Teuchos::sublist(params_, "Rebalance Options");

    // Set the desired parameters. The options are shown in
    // TRILINOS_ROOT/packages/stk/stk_rebalance/ZontanPartition.cpp

//      load_balance_method.set("LOAD BALANCING METHOD"      , "4");
//      load_balance_method.set("ZOLTAN DEBUG LEVEL"      , "10");

      graph_options.sublist(stk::rebalance::Zoltan::default_parameters_name()) = *load_balance_method;

    }

    const Teuchos::MpiComm<int>* mpiComm = dynamic_cast<const Teuchos::MpiComm<int>* > (comm.get());

    stk::rebalance::Zoltan zoltan_partition(*bulkData, *mpiComm->getRawMpiComm(), numDim, graph_options);
    stk::rebalance::rebalance(*bulkData, owned_selector, coordinates_field, NULL, zoltan_partition);

    imbalance = stk::rebalance::check_balance(*bulkData, NULL,
      stk::topology::NODE_RANK, &selector);

    if(comm->getRank() == 0)
      std::cout << "After rebalance: Imbalance threshold is = " << imbalance << endl;


#else
  // Silence compiler warnings
  (void) params_;
  (void) comm;
#endif  //ALBANY_ZOLTAN
}


void GenericSTKMeshStruct::addNodeSetsFromSideSets ()
{
  TEUCHOS_TEST_FOR_EXCEPTION (this->meshSpecs[0]==Teuchos::null, std::runtime_error,
                              "Error! Mesh specs have not been initialized yet.\n");

  // This function adds a (sideset) part to nsPartVec and to the meshSpecs (nsNames)
  for (const auto& ssn_part_pair : ssPartVec)
  {
    // If a nodeset with the same name already exists, we ASSUME it contains this sideset's nodes.
    auto itns = nsPartVec.find(ssn_part_pair.first);
    if (itns!=nsPartVec.end())
      return;

    // Add the part to the node sets parts vector
    stk::mesh::Part* part = ssn_part_pair.second;
    nsPartVec[ssn_part_pair.first] = part;

    // Update the list of nodesets in the mesh specs
    this->meshSpecs[0]->nsNames.push_back(ssn_part_pair.first);

    // This list will be used later to check that the new nodesets' integrity
    m_nodesets_from_sidesets.push_back(ssn_part_pair.first);
  }
}

void GenericSTKMeshStruct::checkNodeSetsFromSideSetsIntegrity ()
{
  // For each nodeset generated from a sideset, this method checks that
  // the sides in the corresponding nodeset contain the right number of
  // nodes, that is, makes sure that 'declare_relation' was called to
  // establish the relation between the side and its node.

  for (auto ssn : m_nodesets_from_sidesets)
  {
    // Fetch the part
    auto it = ssPartVec.find(ssn);
    TEUCHOS_TEST_FOR_EXCEPTION (it==ssPartVec.end(), std::runtime_error,
                                "Error! Side set " << ssn << " not found. This error should NEVER occur though. Bug?\n");
    stk::mesh::Part* ssPart = it->second;

    // Extract sides
    stk::mesh::Selector selector (*ssPart & (metaData->locally_owned_part() | metaData->globally_shared_part()));
    std::vector<stk::mesh::Entity> sides;
    stk::mesh::get_selected_entities(selector, bulkData->buckets(metaData->side_rank()), sides);

    // For each side, we check that it has the right number of nodes.
    unsigned num_nodes = metaData->get_topology(*ssPart).num_nodes();
    for (const auto& side : sides)
    {
      TEUCHOS_TEST_FOR_EXCEPTION (bulkData->num_nodes(side)==num_nodes, std::runtime_error,
                                  "Error! Found a side with wrong number of nodes stored. Most likely,"
                                  "its nodes were not added to the side with 'declare_relation').\n");
    }
  }
}

void GenericSTKMeshStruct::initializeSideSetMeshSpecs (const Teuchos::RCP<const Teuchos_Comm>& comm) {
  // Loop on all mesh specs
  for (auto ms : meshSpecs ) {
    // Loop on all side sets of the mesh
    for (auto ssName : ms->ssNames) {
      // Get the part
      stk::mesh::Part* part = metaData->get_part(ssName);
      TEUCHOS_TEST_FOR_EXCEPTION (part==nullptr, std::runtime_error, "Error! One of the stored meshSpecs claims to have sideset " + ssName +
                                                                     " which, however, is not a part of the mesh.\n");
      stk::topology stk_topo_data = metaData->get_topology( *part );
      shards::CellTopology shards_ctd = stk::mesh::get_cell_topology(stk_topo_data);
      const auto* ctd = shards_ctd.getCellTopologyData();

      auto& ss_ms = ms->sideSetMeshSpecs[ssName];

      // At this point, we cannot assume there will be a discretization on this side set, so we use cubature degree=-1,
      // and the workset size of this mesh. If the user *does* add a discretization (in the Side Set Discretizations sublist),
      // he/she can specify cubature and workset size there. The method initializeSideSetMeshStructs will overwrite
      // this mesh specs anyways.
      // Note: it *may* be that the user need no cubature on this side (only node/cell fields).
      //       But if the user *does* need cubature, we want to set a *very wrong* number, so that
      //       the code will crash somewhere, and he/sh can realize he/she needs to set cubature info somewhere

      // We allow a null ctd here, and we simply do not store the side mesh specs.
      // The reason is that we _may_ be loading a mesh that stores some empty side sets.
      // If we are using the sideset, we will probably run into some sort of errors later,
      // unless we are specifying a side discretization (which will be created _after_ this function call).
      if (ctd==nullptr) {
        Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
        out->setProcRankAndSize(comm->getRank(), comm->getSize());
        out->setOutputToRootOnly(0);
        *out << "Warning! Side set '" << ssName << "' does not store a valid cell topology.\n";

        continue;
      }

      ss_ms.resize(1);
      ss_ms[0] = Teuchos::rcp( new MeshSpecsStruct() );
      ss_ms[0]->ctd = *ctd;
      ss_ms[0]->numDim = this->numDim-1;
    }
  }
}
void GenericSTKMeshStruct::createSideMeshMaps ()
{
  for (auto& it : sideSetMeshStructs) {
    auto ss_stk_mesh = Teuchos::rcp_dynamic_cast<AbstractSTKMeshStruct>(it.second,true);
    auto ss_ms = ss_stk_mesh->meshSpecs[0];

    // We need to create the 2D cell -> (3D cell, side_node_ids) map in the side mesh now
    using ISFT = AbstractSTKFieldContainer::STKIntState;
    ISFT* side_to_cell_map = &ss_stk_mesh->metaData->declare_field<int> (stk::topology::ELEM_RANK, "side_to_cell_map");
    stk::mesh::put_field_on_mesh(*side_to_cell_map, ss_stk_mesh->metaData->universal_part(), 1, nullptr);
#ifdef ALBANY_SEACAS
    stk::io::set_field_role(*side_to_cell_map, Ioss::Field::TRANSIENT);
#endif
    // We need to create the 2D cell -> (3D cell, side_node_ids) map in the side mesh now
    const int num_nodes = ss_ms->ctd.node_count;
    ISFT* side_nodes_ids = &ss_stk_mesh->metaData->declare_field<int> (stk::topology::ELEM_RANK, "side_nodes_ids");
    stk::mesh::put_field_on_mesh(*side_nodes_ids, ss_stk_mesh->metaData->universal_part(), num_nodes, nullptr);
#ifdef ALBANY_SEACAS
    stk::io::set_field_role(*side_nodes_ids, Ioss::Field::TRANSIENT);
#endif
  }
}

void GenericSTKMeshStruct::
buildCellSideNodeNumerationMap (const std::string& sideSetName,
                                std::map<GO,GO>& sideMap,
                                std::map<GO,std::vector<int>>& sideNodeMap)
{
  TEUCHOS_TEST_FOR_EXCEPTION (sideSetMeshStructs.count(sideSetName)==0, Teuchos::Exceptions::InvalidParameter,
      "Error in 'buildCellSideNodeNumerationMap': side set " << sideSetName << " does not have a mesh.\n");

  auto side_mesh = Teuchos::rcp_dynamic_cast<AbstractSTKMeshStruct>(sideSetMeshStructs.at(sideSetName));

  // NOTE 1: the stk fields memorize maps from 2D to 3D values (since fields are defined on 2D mesh), while the input
  //         maps map 3D values to 2D ones. For instance, if node 0 of side3D 41 is mapped to node 2 of cell2D 12, the
  //         stk field values will be field(12) = [*, *, 0], while the input map will be sideNodeMap[41] = [2, *, *]
  //         This is because in Load/SaveSideSetStateField we can only use 3D data to access the map. On the other hand,
  //         the stk fields are defined in 2D, so it makes sense to have them indicized on the 2D values.
  // NOTE 2: The stk side_map maps a 2D cell id to a pair <cell3D_GID, side_lid>, where side_lid is the lid of the side
  //         within the element. The input map, instead, maps the directly side3D_GID into the cell3D_GID..

  // Extract 2D cells
  stk::mesh::Selector selector = stk::mesh::Selector(side_mesh->metaData->locally_owned_part());
  std::vector<stk::mesh::Entity> cells2D;
  stk::mesh::get_selected_entities(selector, side_mesh->bulkData->buckets(stk::topology::ELEM_RANK), cells2D);

  if (cells2D.size()==0) {
    // It can happen if the mesh is partitioned and this process does not own the side
    return;
  }

  const stk::topology::rank_t SIDE_RANK = metaData->side_rank();
  const int num_nodes = side_mesh->bulkData->num_nodes(cells2D[0]);
  int* cell3D_id;
  int* side_nodes_ids;
  GO cell2D_GID, side3D_GID;
  int side_lid;
  int num_sides;
  using ISFT = AbstractSTKFieldContainer::STKIntState;
  ISFT* side_to_cell_map   = side_mesh->metaData->get_field<int> (stk::topology::ELEM_RANK, "side_to_cell_map");
  ISFT* side_nodes_ids_map = side_mesh->metaData->get_field<int> (stk::topology::ELEM_RANK, "side_nodes_ids");
  std::vector<stk::mesh::EntityId> cell2D_nodes_ids(num_nodes), side3D_nodes_ids(num_nodes);
  const stk::mesh::Entity* side3D_nodes;
  const stk::mesh::Entity* cell2D_nodes;
  if (side_mesh->side_maps_present && !side_mesh->ignore_side_maps) {
    // This mesh was loaded from a file that stored the side maps.
    // Hence, we just read it and stuff the map with it
    // WARNING: the maps may be not be valid. This can happen if they were built with an
    //          extruded mesh with N layers, and we are now using this side mesh to build
    //          an extruded mesh with M!=N layers, or with different ordering (COLUMN/LAYER).
    //          If this is the case, you must first edit the exodus file and delete the maps
    //          or set 'Ignore Side Maps' to true in the input file
    for (const auto& cell2D : cells2D) {
      // Get the stk field data
      cell3D_id      = stk::mesh::field_data(*side_to_cell_map, cell2D);
      side_nodes_ids = stk::mesh::field_data(*side_nodes_ids_map, cell2D);
      stk::mesh::Entity cell3D = bulkData->get_entity(stk::topology::ELEM_RANK,*cell3D_id);

      num_sides = bulkData->num_sides(cell3D);
      const stk::mesh::Entity* cell_sides = bulkData->begin(cell3D,SIDE_RANK);
      side_lid = -1;
      for (int iside(0); iside<num_sides; ++iside) {
        side3D_nodes = bulkData->begin_nodes(cell_sides[iside]);
        for (int inode(0); inode<num_nodes; ++inode) {
          side3D_nodes_ids[inode] = bulkData->identifier(side3D_nodes[inode]);
        }
        if (std::is_permutation(side3D_nodes_ids.begin(),side3D_nodes_ids.end(), side_nodes_ids)) {
          side_lid = iside;
          side3D_GID = bulkData->identifier(cell_sides[iside])-1;
          break;
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION (side_lid==-1, std::logic_error, "Error! Cannot locate the right side in the cell.\n");

      sideMap[side3D_GID] = side_mesh->bulkData->identifier(cell2D)-1;
      sideNodeMap[side3D_GID].resize(num_nodes);
      cell2D_nodes = side_mesh->bulkData->begin_nodes(cell2D);
      for (int i(0); i<num_nodes; ++i) {
        auto it = std::find(side3D_nodes_ids.begin(),side3D_nodes_ids.end(),side_nodes_ids[i]);
        sideNodeMap[side3D_GID][std::distance(side3D_nodes_ids.begin(),it)] = i;
      }
    }

    return;
  }

  const stk::mesh::Entity* side_cells;
  for (const auto& cell2D : cells2D) {
    // Get the stk field data
    cell3D_id      = stk::mesh::field_data(*side_to_cell_map, cell2D);
    side_nodes_ids = stk::mesh::field_data(*side_nodes_ids_map, cell2D);

    // The side-id is assumed equal to the cell-id in the side mesh...
    side3D_GID = cell2D_GID = side_mesh->bulkData->identifier(cell2D)-1;
    stk::mesh::Entity side3D = bulkData->get_entity(SIDE_RANK, side3D_GID+1);

    // Safety check
    TEUCHOS_TEST_FOR_EXCEPTION (bulkData->num_elements(side3D)!=1, std::logic_error,
                                "Error! Side " << side3D_GID << " has more/less than 1 adjacent element.\n");

    side_cells = bulkData->begin_elements(side3D);
    stk::mesh::Entity cell3D = side_cells[0];

    *cell3D_id = bulkData->identifier(cell3D);

    sideMap[side3D_GID] = cell2D_GID;
    sideNodeMap[side3D_GID].resize(num_nodes);

    // Now we determine the lid of the side within the element and also the node ordering
    cell2D_nodes = side_mesh->bulkData->begin_nodes(cell2D);
    side3D_nodes = bulkData->begin_nodes(side3D);
    for (int i(0); i<num_nodes; ++i) {
      cell2D_nodes_ids[i] = side_mesh->bulkData->identifier(cell2D_nodes[i]);
      side3D_nodes_ids[i] = bulkData->identifier(side3D_nodes[i]);
    }

    for (int i(0); i<num_nodes; ++i) {
      auto it = std::find(cell2D_nodes_ids.begin(),cell2D_nodes_ids.end(),side3D_nodes_ids[i]);
      sideNodeMap[side3D_GID][i] = std::distance(cell2D_nodes_ids.begin(),it);
      side_nodes_ids[std::distance(cell2D_nodes_ids.begin(),it)] = side3D_nodes_ids[i];
    }
  }

  // Just in case this method gets called twice
  side_mesh->side_maps_present = true;
}

void GenericSTKMeshStruct::printParts(stk::mesh::MetaData *metaData)
{
  std::cout << "Printing all part names of the parts found in the metaData:" << std::endl;
  stk::mesh::PartVector all_parts = metaData->get_parts();

  for (auto i_part = all_parts.begin(); i_part != all_parts.end(); ++i_part) {
    stk::mesh::Part* part = *i_part ;
    std::cout << "\t" << part->name() << std::endl;
  }
}

void GenericSTKMeshStruct::
loadRequiredInputFields (const Teuchos::RCP<const Teuchos_Comm>& comm)
{
  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
  out->setProcRankAndSize(comm->getRank(), comm->getSize());
  out->setOutputToRootOnly(0);

  *out << "[GenericSTKMeshStruct] Processing field requirements...\n";

  // Load required fields
  stk::mesh::Selector select_owned_in_part = stk::mesh::Selector(metaData->universal_part()) & stk::mesh::Selector(metaData->locally_owned_part());

  stk::mesh::Selector select_overlap_in_part = stk::mesh::Selector(metaData->universal_part()) & (stk::mesh::Selector(metaData->locally_owned_part()) | stk::mesh::Selector(metaData->globally_shared_part()));

  std::vector<stk::mesh::Entity> nodes, elems;
  stk::mesh::get_selected_entities(select_overlap_in_part, bulkData->buckets(stk::topology::NODE_RANK), nodes);
  stk::mesh::get_selected_entities(select_owned_in_part, bulkData->buckets(stk::topology::ELEM_RANK), elems);

  Teuchos::Array<GO> nodeIndices(nodes.size()), elemIndices(elems.size());
  for (unsigned int i = 0; i < nodes.size(); ++i) {
    nodeIndices[i] = bulkData->identifier(nodes[i]) - 1;
  }
  for (unsigned int i = 0; i < elems.size(); ++i) {
    elemIndices[i] = bulkData->identifier(elems[i]) - 1;
  }

  auto nodes_vs = createVectorSpace(comm,nodeIndices);
  auto elems_vs = createVectorSpace(comm,elemIndices);

  // Check whether we need the serial map or not. The only scenario where we DO need it is if we are
  // loading a field from an ASCII file. So let's check the fields info to see if that's the case.
  Teuchos::ParameterList dummyList;
  Teuchos::ParameterList* req_fields_info;
  if (params->isSublist("Required Fields Info")) {
    req_fields_info = &params->sublist("Required Fields Info");
  } else {
    req_fields_info = &dummyList;
  }
  int num_fields = req_fields_info->get<int>("Number Of Fields",0);
  bool node_field_ascii_loads = false;
  bool elem_field_ascii_loads = false;
  std::string fname, fusage, ftype, forigin;
  for (int ifield=0; ifield<num_fields; ++ifield) {
    std::stringstream ss;
    ss << "Field " << ifield;
    Teuchos::ParameterList& fparams = req_fields_info->sublist(ss.str());

    fusage = fparams.get<std::string>("Field Usage", "Input");
    ftype  = fparams.get<std::string>("Field Type","INVALID");
    if (fusage == "Input" || fusage == "Input-Output") {
      forigin = fparams.get<std::string>("Field Origin","INVALID");
      if (forigin=="File" && fparams.isParameter("File Name")) {
        if (ftype.find("Node")!=std::string::npos) {
          node_field_ascii_loads = true;
        } else if (ftype.find("Elem")!=std::string::npos) {
          elem_field_ascii_loads = true;
        }
      }
    }
  }

  // NOTE: the serial vs cannot be created linearly, with GIDs from 0 to numGlobalNodes/Elems, since
  //       this may be a boundary mesh, and the GIDs may not start from 0, nor be contiguous.
  //       Therefore, we must create a root vs. Moreover, we need the GIDs sorted (so that, regardless
  //       of the GID, we read the serial input files in the correct order), and we can't sort them
  //       once the vs is created.

  auto serial_nodes_vs = nodes_vs;
  auto serial_elems_vs = elems_vs;
  if (node_field_ascii_loads) {
    Teuchos::Array<GO> nodes_gids = getGlobalElements(nodes_vs);
    Teuchos::Array<GO> all_nodes_gids;
    gatherV(comm,nodes_gids(),all_nodes_gids,0);
    std::sort(all_nodes_gids.begin(),all_nodes_gids.end());
    auto it = std::unique(all_nodes_gids.begin(),all_nodes_gids.end());
    all_nodes_gids.erase(it,all_nodes_gids.end());
    serial_nodes_vs = createVectorSpace(comm,all_nodes_gids);
  }
  if (elem_field_ascii_loads) {
    Teuchos::Array<GO> elems_gids = getGlobalElements(elems_vs);
    Teuchos::Array<GO> all_elems_gids;
    gatherV(comm,elems_gids(),all_elems_gids,0);
    std::sort(all_elems_gids.begin(),all_elems_gids.end());
    serial_elems_vs = createVectorSpace(comm,all_elems_gids);
  }

  // Creating the combine and scatter manager object (to transfer from serial to parallel vectors)
  auto cas_manager_node = createCombineAndScatterManager(serial_nodes_vs,nodes_vs);
  auto cas_manager_elem = createCombineAndScatterManager(serial_elems_vs,elems_vs);

  for (int ifield=0; ifield<num_fields; ++ifield) {
    std::stringstream ss;
    ss << "Field " << ifield;
    Teuchos::ParameterList& fparams = req_fields_info->sublist(ss.str());

    // First, get the name and usage of the field, and check if it's used
    if (fparams.isParameter("State Name")) {
      fname = fparams.get<std::string>("State Name");
    } else {
      fname = fparams.get<std::string>("Field Name");
    }
    fusage = fparams.get<std::string>("Field Usage", "Input");
    if (fusage == "Unused") {
      *out << "  - Skipping field '" << fname << "' since it's listed as unused.\n";
      continue;
    }

    // The field is used somehow. Check that it is present in the mesh
    ftype = fparams.get<std::string>("Field Type","INVALID");
    checkFieldIsInMesh(fname, ftype);

    // Check if it's an output file (nothing to be done then). If not, check that the usage is a valid string
    if (fusage == "Output") {
      *out << "  - Skipping field '" << fname << "' since it's listed as output. Make sure there's an evaluator set to save it!\n";
      continue;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (fusage!="Input" && fusage!="Input-Output", Teuchos::Exceptions::InvalidParameter,
                                  "Error! 'Field Usage' for field '" << fname << "' must be one of 'Input', 'Output', 'Input-Output' or 'Unused'.\n");
    }

    // Ok, it's an input (or input-output) field. Find out where the field comes from
    forigin = fparams.get<std::string>("Field Origin","INVALID");
    if (forigin=="Mesh") {
      *out << "  - Skipping field '" << fname << "' since it's listed as present in the mesh.\n";
      continue;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (forigin!="File", Teuchos::Exceptions::InvalidParameter,
                                  "Error! 'Field Origin' for field '" << fname << "' must be one of 'File' or 'Mesh'.\n");
    }

    // The field is not already present (with updated values) in the mesh, and must be loaded/computed filled here.

    // Detect load type
    bool load_ascii = fparams.isParameter("File Name");
    bool load_math_expr = fparams.isParameter("Field Expression");
    bool load_value = fparams.isParameter("Field Value") || fparams.isParameter("Random Value");
    TEUCHOS_TEST_FOR_EXCEPTION ( load_ascii && load_math_expr, std::logic_error, "Error! You cannot specify both 'File Name' and 'Field Expression' for loading a field.\n");
    TEUCHOS_TEST_FOR_EXCEPTION ( load_ascii && load_value,     std::logic_error, "Error! You cannot specify both 'File Name' and 'Field Value' (or 'Random Value') for loading a field.\n");
    TEUCHOS_TEST_FOR_EXCEPTION ( load_math_expr && load_value, std::logic_error, "Error! You cannot specify both 'Field Expression' and 'Field Value' (or 'Random Value') for loading a field.\n");

    // Depending on the input field type, we need to use different pointers/importers/vectors
    bool nodal, scalar, layered;
    Teuchos::RCP<CombineAndScatterManager> cas_manager;
    std::vector<stk::mesh::Entity>* entities;
    if (ftype == "Node Scalar") {
      nodal = true; scalar = true; layered = false;
      cas_manager = cas_manager_node;
      entities = &nodes;
    } else if (ftype == "Elem Scalar") {
      nodal = false; scalar = true; layered = false;
      cas_manager = cas_manager_elem;
      entities = &elems;
    } else if (ftype == "Node Vector") {
      nodal = true; scalar = false; layered = false;
      cas_manager = cas_manager_node;
      entities = &nodes;
    } else if (ftype == "Elem Vector") {
      nodal = false; scalar = false; layered = false;
      cas_manager = cas_manager_elem;
      entities = &elems;
    } else if (ftype == "Node Layered Scalar") {
      nodal = true; scalar = true; layered = true;
      cas_manager = cas_manager_node;
      entities = &nodes;
    } else if (ftype == "Elem Layered Scalar") {
      nodal = false; scalar = true; layered = true;
      cas_manager = cas_manager_elem;
      entities = &elems;
    } else if (ftype == "Node Layered Vector") {
      nodal = true; scalar = false; layered = true;
      cas_manager = cas_manager_node;
      entities = &nodes;
    } else if (ftype == "Elem Layered Vector") {
      nodal = false; scalar = false; layered = true;
      cas_manager = cas_manager_elem;
      entities = &elems;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameterValue,
                                  "Error! Field '" << fname << "' has type '" << ftype << "'.\n" <<
                                  "Unfortunately, the only supported field types so fare are 'Node/Elem Scalar/Vector' and 'Node/Elem Layered Scalar/Vector'.\n");
    }

    auto serial_vs = cas_manager->getOwnedVectorSpace();
    auto vs = cas_manager->getOverlappedVectorSpace();  // It is not overlapped, it is just distributed.
    Teuchos::RCP<Thyra_MultiVector> field_mv;

    std::vector<double> dummy;
    auto& norm_layers_coords = layered ? fieldContainer->getMeshVectorStates()[fname + "_NLC"] : dummy;
    if (load_ascii) {
      field_mv = loadField (fname, fparams, *cas_manager, comm, nodal, scalar, layered, out, norm_layers_coords);
    } else if (load_value) {
      field_mv = fillField (fname, fparams, vs, nodal, scalar, layered, out, norm_layers_coords);
    } else if (load_math_expr) {
      field_mv = computeField (fname, fparams, vs, *entities, nodal, scalar, layered, out);
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! No means were specified for loading field '" + fname + "'.\n");
    }

    auto field_mv_view = getLocalData(field_mv.getConst());

    //Now we have to stuff the vector in the mesh data
    using SFT = AbstractSTKFieldContainer::STKFieldType;
    stk::topology::rank_t entity_rank = nodal ? stk::topology::NODE_RANK : stk::topology::ELEM_RANK;
    SFT* stk_field = metaData->get_field<double> (entity_rank, fname);
    TEUCHOS_TEST_FOR_EXCEPTION (stk_field==nullptr, std::logic_error,
                                "Error! Field " << fname << " not present (perhaps is not '" << ftype << "'?).\n");

    stk::mesh::EntityId gid;
    LO lid;
    auto indexer = createGlobalLocalIndexer(vs);
    for (unsigned int i(0); i<entities->size(); ++i) {
      double* values = stk::mesh::field_data(*stk_field, (*entities)[i]);

      gid = bulkData->identifier((*entities)[i]) - 1;
      lid = indexer->getLocalElement(GO(gid));
      for (int iDim(0); iDim<field_mv_view.size(); ++iDim) {
        values[iDim] = field_mv_view[iDim][lid];
      }
    }
  }
}

Teuchos::RCP<Thyra_MultiVector>
GenericSTKMeshStruct::
computeField (const std::string& field_name,
              const Teuchos::ParameterList& field_params,
              const Teuchos::RCP<const Thyra_VectorSpace>& entities_vs,
              const std::vector<stk::mesh::Entity>& entities,
              bool nodal, bool scalar, bool layered,
              const Teuchos::RCP<Teuchos::FancyOStream> out)
{
#ifdef ALBANY_PANZER_EXPR_EVAL
  // Only nodal fields allowed, no layered fields
  TEUCHOS_TEST_FOR_EXCEPTION(!nodal, std::logic_error, "Error! Only nodal fields can be computed from a mathematical expression.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(layered, std::logic_error, "Error! Layered fields cannot be computed from a mathematical expression.\n");

  int field_dim = 1;
  if (!scalar) {
    TEUCHOS_TEST_FOR_EXCEPTION(!field_params.isParameter("Vector Dim"), std::logic_error,
                               "Error! In order to compute the vector field '" << field_name << "' "
                               "from a mathematical expression, you must provide the parameter 'Vector Dim'.\n");
    field_dim = field_params.get<int>("Vector Dim");
  }

  // Get the expressions out of the parameter list.
  Teuchos::Array<std::string> expressions = field_params.get<Teuchos::Array<std::string>>("Field Expression");

  // NOTE: we need expressions to be of length AT LEAST equal to the field dimension.
  //       If the length L is larger than the field dimension M, then the first L-M
  //       strings are assumed to be coefficients needed for the field formula.
  //       E.g.: if we have a field of dimension 2, one could write
  //         <Parameter name="Field Expression" type="Array(string)" value="{a=1.5;b=-1;c=2;a*x^2+b*x+c;a*x+b*x+c}"/>

  int num_expr = expressions.size();
  std::string temp_str;
  std::string field_type = (nodal ? "Node" : "Elem");
  field_type += (layered ? " Layered" : "");
  field_type += (scalar ? " Scalar" : " Vector");
  TEUCHOS_TEST_FOR_EXCEPTION(num_expr<field_dim, Teuchos::Exceptions::InvalidParameter,
                             "Error! Input array for 'Field Expression' is too short. "
                             "Expected length >=" << field_dim << ". Got " << num_expr << " instead.\n");

  *out << "  - Computing " << field_type << " field '" << field_name << "' from mathematical expression(s):";
  int num_expr_params = num_expr - field_dim;
  for (int idim=num_expr_params; idim<num_expr; ++idim) {
    *out << " " << expressions[idim] << (idim==num_expr-1 ? "" : ";");
  }
  if (num_expr_params>0) {
    *out << " (with";
    for (int idim=0; idim<num_expr_params; ++idim) {
      *out << " " << expressions[idim] << (idim==num_expr_params-1 ? "" : ";");
    }
    *out << ")";
  }
  *out << ".\n";

  // Extract coordinates of all nodes
  auto field_mv = Thyra::createMembers(entities_vs,field_dim);
  using exec_space = Tpetra_MultiVector::execution_space;
  using view_type = Kokkos::View<double**,DeviceView1d<double>::memory_space>;
  using layout = view_type::traits::array_layout;

  view_type x("x",entities.size(),1), y("y",entities.size(),1), z("z",entities.size(),1);
  view_type::HostMirror x_h = Kokkos::create_mirror_view(x);
  view_type::HostMirror y_h = Kokkos::create_mirror_view(y);
  view_type::HostMirror z_h = Kokkos::create_mirror_view(z);
  const auto& coordinates = *this->getCoordinatesField3d();
  double* xyz;
  for (unsigned int i=0; i<entities.size(); ++i) {
    xyz = stk::mesh::field_data(coordinates, entities[i]);

    x_h(i,0) = xyz[0];
    y_h(i,0) = xyz[1];
    z_h(i,0) = xyz[2];
  }
  Kokkos::deep_copy(x,x_h);
  Kokkos::deep_copy(y,y_h);
  Kokkos::deep_copy(z,z_h);

  // Set up the expression parser
  panzer::Expr::Eval<double**,layout,exec_space> eval;
  using const_view_type = decltype(eval)::const_view_type;
  set_cmath_functions(eval);
  eval.set("x",x);
  eval.set("y",y);
  eval.set("z",z);

  // Start by reading the parameters used in the field expression(s)
  Teuchos::any result;
  for (int iparam=0; iparam<num_expr_params; ++iparam) {
    eval.read_string(result,expressions[iparam]+";","params");
  }

  // Parse and evaluate all the expressions
  for (int idim=0; idim<field_dim; ++idim) {
    eval.read_string(result,expressions[num_expr_params+idim],"field expression");
    auto result_view = Teuchos::any_cast<const_view_type>(result);
    auto result_view_1d = DeviceView1d<const double>(result_view.data(),result_view.extent_int(0));
    Kokkos::deep_copy(getNonconstDeviceData(field_mv->col(idim)),result_view_1d);
  }
#else
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Error! Cannot read the field from a mathematical expression, since PanzerExprEval package was not found in Trilinos.\n");
#endif

  return field_mv;
}

void GenericSTKMeshStruct::checkFieldIsInMesh (const std::string& fname, const std::string& ftype) const
{
  stk::topology::rank_t entity_rank;
  if (ftype.find("Node")==std::string::npos) {
    entity_rank = stk::topology::ELEM_RANK;
  } else {
    entity_rank = stk::topology::NODE_RANK;
  }

  bool missing = (metaData->get_field<double> (entity_rank, fname)==nullptr);

  if (missing) {
    TEUCHOS_TEST_FOR_EXCEPTION (
        missing, std::runtime_error,
        "Error! The field '" << fname << "' was not found in the mesh.\n"
        "  Probably it was not registered it in the state manager (which forwards it to the mesh)\n");
  }
}

Teuchos::RCP<Teuchos::ParameterList>
GenericSTKMeshStruct::getValidGenericSTKParameters(std::string listname) const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList(listname));;
  validPL->set<std::string>("Cell Topology", "Quad" , "Quad or Tri Cell Topology");
  validPL->set<int>("Number Of Time Derivatives", 1, "Number of time derivatives in use in the problem");
  validPL->set<std::string>("Exodus Output File Name", "",
      "Request exodus output to given file name. Requires SEACAS build");
  validPL->set<std::string>("Exodus Solution Name", "",
      "Name of solution output vector written to Exodus file. Requires SEACAS build");
  validPL->set<std::string>("Exodus SolutionDot Name", "",
      "Name of solution output vector written to Exodus file. Requires SEACAS build");
  validPL->set<std::string>("Exodus SolutionDotDot Name", "",
      "Name of solution output vector written to Exodus file. Requires SEACAS build");
  validPL->set<std::string>("Exodus Residual Name", "",
      "Name of residual output vector written to Exodus file. Requires SEACAS build");
#ifdef ALBANY_DTK
  validPL->set<std::string>("Exodus Solution DTK Name", "",
      "Name of solution dtk written to Exodus file. Requires SEACAS build");
  validPL->set<std::string>("Exodus SolutionDot DTK Name", "",
      "Name of solution_dot dtk written to Exodus file. Requires SEACAS build");
  validPL->set<std::string>("Exodus SolutionDotDot DTK Name", "",
      "Name of solution_dotdot dtk written to Exodus file. Requires SEACAS build");
#endif
  validPL->set<bool>("Output DTK Field to Exodus", true, "Boolean indicating whether to write dtk field to exodus file");
  validPL->set<int>("Exodus Write Interval", 3, "Step interval to write solution data to Exodus file");
  validPL->set<std::string>("Method", "",
    "The discretization method, parsed in the Discretization Factory");

  validPL->set<std::string>("Cubature Rule", "", "Integration rule sent to Intrepid2: GAUSS, GAUSS_RADAU_LEFT, GAUSS_RADAU_RIGHT, GAUSS_LOBATTO");
  validPL->set<int>("Workset Size", DEFAULT_WORKSET_SIZE, "Upper bound on workset (bucket) size");
  validPL->set<bool>("Use Automatic Aura", false, "Use automatic aura with BulkData");
  validPL->set<bool>("Separate Evaluators by Element Block", false,
                     "Flag for different evaluation trees for each Element Block");
  validPL->set<std::string>("Transform Type", "None", "None or ISMIP-HOM Test A"); //for LandIce problem that require transformation of STK mesh
  validPL->set<int>("Element Degree", 1, "Element degree (points per edge - 1) in enriched Aeras mesh");
  validPL->set<bool>("Write Coordinates to MatrixMarket", false, "Writing Coordinates to MatrixMarket File"); //for writing coordinates to matrix market file
  validPL->set<double>("LandIce alpha", 0.0, "Surface boundary inclination for LandIce problems (in degrees)"); //for LandIce problem that require transformation of STK mesh
  validPL->set<double>("LandIce L", 1, "Domain length for LandIce problems"); //for LandIce problem that require transformation of STK mesh

  validPL->set<double>("x-shift", 0.0, "Value by which to shift domain in positive x-direction");
  validPL->set<double>("y-shift", 0.0, "Value by which to shift domain in positive y-direction");
  validPL->set<double>("z-shift", 0.0, "Value by which to shift domain in positive z-direction");
  validPL->set<Teuchos::Array<double>>("Betas BL Transform", Teuchos::tuple<double>(0.0, 0.0, 0.0), "Beta parameters for Tanh Boundary Layer transform type");

  validPL->set<bool>("Contiguous IDs", true, "Tells Ascii mesh reader is mesh has contiguous global IDs on 1 processor."); //for LandIce problem that require transformation of STK mesh
  validPL->set<bool>("Save Solution Field", true, "Whether the solution field should be saved in the output mesh");

  Teuchos::Array<std::string> defaultFields;
  validPL->set<Teuchos::Array<std::string> >("Restart Fields", defaultFields,
                     "Fields to pick up from the restart file when restarting");
  validPL->set<Teuchos::Array<std::string> >("Solution Vector Components", defaultFields,
      "Names and layout of solution output vector written to Exodus file. Requires SEACAS build");
  validPL->set<Teuchos::Array<std::string> >("SolutionDot Vector Components", defaultFields,
      "Names and layout of solution_dot output vector written to Exodus file. Requires SEACAS build");
  validPL->set<Teuchos::Array<std::string> >("SolutionDotDot Vector Components", defaultFields,
      "Names and layout of solution_dotdot output vector written to Exodus file. Requires SEACAS build");
  validPL->set<Teuchos::Array<std::string> >("Residual Vector Components", defaultFields,
      "Names and layout of residual output vector written to Exodus file. Requires SEACAS build");

  validPL->set<bool>("Use Serial Mesh", false, "Read in a single mesh on PE 0 and rebalance");
  validPL->set<bool>("Transfer Solution to Coordinates", false, "Copies the solution vector to the coordinates for output");
  validPL->set<bool>("Set All Parts IO", false, "If true, all parts are marked as io parts");
  validPL->set<bool>("Build Node Sets From Side Sets",false,"Flag to build node sets from side sets");
  validPL->set<bool>("Export 3d coordinates field",false,"If true AND the mesh dimension is not already 3, export a 3d version of the coordinate field.");

  validPL->sublist("Required Fields Info", false, "Info for the creation of the required fields in the STK mesh");

  validPL->set<bool>("Ignore Side Maps", true, "If true, we ignore possible side maps already imported from the exodus file");

// Uniform percept adaptation of input mesh prior to simulation
  validPL->set<bool>("Rebalance Mesh", false, "Parallel re-load balance initial mesh after generation");

  validPL->sublist("Side Set Discretizations", false, "A sublist containing info for storing side discretizations");
 
  //The following are for observing dxdp and dgdp for transient sensitivities 
  validPL->set<std::string>("Sensitivity Method", "None", "Type of sensitivities requested");
  validPL->set<int>("Response Function Index", 0, "Response function index (for adjoint transient sensitivities)");
  validPL->set<int>("Sensitivity Parameter Index", 0, "Parameter sensitivity index (for transient sensitivities)");
  validPL->sublist("Mesh Adaptivity", false, "Sublist containing mesh adaptivity options");

  return validPL;
}

} // namespace Albany
