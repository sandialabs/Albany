//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <stdio.h>
#include <unistd.h>
#include <iostream>

#include "Albany_ExtrudedSTKMeshStruct.hpp"
#include "Teuchos_VerboseObject.hpp"

#include <Shards_BasicTopologies.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Selector.hpp>

#include <Albany_STKNodeSharing.hpp>

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

#include <boost/algorithm/string/predicate.hpp>

#include "Albany_Utils.hpp"

const Tpetra::global_size_t INVALID = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid ();

Albany::ExtrudedSTKMeshStruct::ExtrudedSTKMeshStruct(const Teuchos::RCP<Teuchos::ParameterList>& params,
                                                     const Teuchos::RCP<const Teuchos_Comm>& comm,
                                                     Teuchos::RCP<Albany::AbstractMeshStruct> inputBasalMesh,
						     const int numParams) :
    GenericSTKMeshStruct(params, 3, numParams), out(Teuchos::VerboseObjectBase::getDefaultOStream()), periodic(false)
{
  params->validateParameters(*getValidDiscretizationParameters(), 0);


  std::vector<std::string> nsNames;
  std::string nsn = "lateral";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = &metaData->declare_part(nsn, stk::topology::NODE_RANK);
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn = "bottom";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = &metaData->declare_part(nsn, stk::topology::NODE_RANK);
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn = "top";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = &metaData->declare_part(nsn, stk::topology::NODE_RANK);
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif

  std::vector<std::string> ssNames;
  std::string ssnLat = "lateralside";
  std::string ssnBottom = "basalside";
  std::string ssnTop = "upperside";

  ssNames.push_back(ssnLat);
  ssNames.push_back(ssnBottom);
  ssNames.push_back(ssnTop);
  std::map<std::string, stk::mesh::Part*> ssPartVecLateral;
  ssPartVecLateral[ssnLat] = &metaData->declare_part(ssnLat, metaData->side_rank());
  ssPartVec[ssnBottom] = &metaData->declare_part(ssnBottom, metaData->side_rank());
  ssPartVec[ssnTop] = &metaData->declare_part(ssnTop, metaData->side_rank());
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*ssPartVecLateral[ssnLat]);
  stk::io::put_io_part_attribute(*ssPartVec[ssnBottom]);
  stk::io::put_io_part_attribute(*ssPartVec[ssnTop]);
#endif

  basalMeshStruct = Teuchos::rcp_dynamic_cast<Albany::AbstractSTKMeshStruct>(inputBasalMesh,false);
  TEUCHOS_TEST_FOR_EXCEPTION (basalMeshStruct==Teuchos::null, std::runtime_error, "Error! Could not cast basal mesh to AbstractSTKMeshStruct.\n");

  stk::mesh::MetaData& metaData2D = *basalMeshStruct->metaData; //bulkData2D.mesh_meta_data();
  auto partvec = metaData2D.get_mesh_parts();
  for (auto part:partvec) {
    if(part->primary_entity_rank() == metaData2D.side_rank()) {
      std::string partName = "extruded_"+part->name();
      ssNames.push_back(partName);
      ssPartVecLateral[partName] = &metaData->declare_part(partName, metaData->side_rank());
#ifdef ALBANY_SEACAS
      stk::io::put_io_part_attribute(*ssPartVecLateral[partName]);
#endif
    }
    if(part->primary_entity_rank() == stk::topology::NODE_RANK) {
      std::string partName = "extruded_"+part->name();
      nsNames.push_back(partName);
      nsPartVec[partName] = &metaData->declare_part(partName, stk::topology::NODE_RANK);
#ifdef ALBANY_SEACAS
      stk::io::put_io_part_attribute(*nsPartVec[partName]);
#endif
      partName = "basal_"+part->name();
      nsNames.push_back(partName);
      nsPartVec[partName] = &metaData->declare_part(partName, stk::topology::NODE_RANK);
#ifdef ALBANY_SEACAS
      stk::io::put_io_part_attribute(*nsPartVec[partName]);
#endif
    }
  }

  for(auto it:ssPartVecLateral)
    ssPartVec[it.first] = it.second;

  sideSetMeshStructs["basalside"] = basalMeshStruct;

  std::string shape = params->get("Element Shape", "Hexahedron");
  std::string basalside_elem_name;
  if(shape == "Tetrahedron")  {
    ElemShape = Tetrahedron;
    basalside_elem_name = shards::getCellTopologyData<shards::Triangle<3> >()->name;
  }
  else if (shape == "Wedge")  {
    ElemShape = Wedge;
    basalside_elem_name = shards::getCellTopologyData<shards::Triangle<3> >()->name;
  }
  else if (shape == "Hexahedron") {
    ElemShape = Hexahedron;
    basalside_elem_name = shards::getCellTopologyData<shards::Quadrilateral<4> >()->name;
  }
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameterValue,
              std::endl << "Error in ExtrudedSTKMeshStruct: Element Shape " << shape << " not recognized. Possible values: Tetrahedron, Wedge, Hexahedron");

  const auto& basalMeshSpec = basalMeshStruct->getMeshSpecs()[0];
  std::string elem2d_name(basalMeshSpec->ctd.base->name);
  TEUCHOS_TEST_FOR_EXCEPTION(basalside_elem_name != elem2d_name, Teuchos::Exceptions::InvalidParameterValue,
                std::endl << "Error in ExtrudedSTKMeshStruct: Expecting topology name of elements of 2d mesh to be " <<  basalside_elem_name << " but it is " << elem2d_name);

  stk::topology etopology;

  switch (ElemShape) {
  case Tetrahedron:
    etopology = stk::topology::TET_4;
    stk::mesh::set_topology(*ssPartVec[ssnBottom], stk::topology::TRI_3);
    stk::mesh::set_topology(*ssPartVec[ssnTop], stk::topology::TRI_3);
    for (auto it:ssPartVecLateral)
      stk::mesh::set_topology(*it.second, stk::topology::TRI_3);
    NumBaseElemeNodes = 3;
    break;
  case Wedge:
    etopology = stk::topology::WEDGE_6;
    stk::mesh::set_topology(*ssPartVec[ssnBottom], stk::topology::TRI_3);
    stk::mesh::set_topology(*ssPartVec[ssnTop], stk::topology::TRI_3);
    for (auto it:ssPartVecLateral)
      stk::mesh::set_topology(*it.second, stk::topology::QUAD_4);
    NumBaseElemeNodes = 3;
    break;
  case Hexahedron:
    etopology = stk::topology::HEX_8;
    stk::mesh::set_topology(*ssPartVec[ssnBottom], stk::topology::QUAD_4);
    stk::mesh::set_topology(*ssPartVec[ssnTop], stk::topology::QUAD_4);
    for (auto it:ssPartVecLateral)
      stk::mesh::set_topology(*it.second, stk::topology::QUAD_4);
    NumBaseElemeNodes = 4;
    break;
  }

  std::string ebn = "Element Block 0";
  partVec.push_back(&metaData->declare_part_with_topology(ebn, etopology));
  shards::CellTopology shards_ctd = stk::mesh::get_cell_topology(etopology);
  this->addElementBlockInfo(0, ebn, partVec[0], shards_ctd);

#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*partVec[0]);
#endif

  numDim = 3;
  numLayers = params->get<int>("NumLayers");
  Ordering = params->get("Columnwise Ordering", false) ? LayeredMeshOrdering::COLUMN : LayeredMeshOrdering::LAYER;

  int cub = params->get("Cubature Degree", 3);
  int basalWorksetSize = basalMeshSpec->worksetSize;
  int worksetSizeMax = params->get<int>("Workset Size", DEFAULT_WORKSET_SIZE);
  int numElemsInColumn = numLayers*((ElemShape==Tetrahedron) ? 3 : 1);
  int ebSizeMaxEstimate = basalWorksetSize * numElemsInColumn; // This is ebSizeMax when basalWorksetSize is max
  int worksetSize = this->computeWorksetSize(worksetSizeMax, ebSizeMaxEstimate);

  const CellTopologyData& ctd = *shards_ctd.getCellTopologyData(); 

  this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub, nsNames, ssNames, worksetSize, 
     ebn, ebNameToIndex, this->interleavedOrdering));
  if (basalMeshSpec->singleWorksetSizeAllocation && worksetSize == ebSizeMaxEstimate)
    this->meshSpecs[0]->singleWorksetSizeAllocation = true;

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

  // Get upper bound on lateral/upper workset sizes by using Ioss element counts on side blocks
  if (basalMeshSpec->singleWorksetSizeAllocation) {
    // Set lateral workset sizes based on basal sidesets
    for (auto bssName : basalMeshSpec->ssNames) {
      // Get maximum workset size of basalside sideset
      const auto& basalSideSetMeshSpecs = basalMeshSpec->sideSetMeshSpecs;
      const auto basalSideSetMeshSpecIter = basalSideSetMeshSpecs.find(bssName);
      TEUCHOS_TEST_FOR_EXCEPTION(basalSideSetMeshSpecIter == basalSideSetMeshSpecs.end(), std::runtime_error,
          "Cannot find " << bssName << " in basalside sideSetMeshSpecs!\n");
      if (!basalSideSetMeshSpecIter->second[0]->singleWorksetSizeAllocation) continue;
      const auto basalSideSetWorksetSize = basalSideSetMeshSpecIter->second[0]->worksetSize;

      // Compute maximum workset size for lateral sideset
      const int num_cells_per_side = ElemShape == Tetrahedron ? 2 : 1;
      int lateralSidesetWorksetSizeMax = num_cells_per_side * basalSideSetWorksetSize * numLayers;

      // Set workset size for lateral sideset to maximum workset size
      const std::string ssName = "extruded_" + bssName;
      const auto& sideSetMeshSpecs = this->meshSpecs[0]->sideSetMeshSpecs;
      auto sideSetMeshSpecIter = sideSetMeshSpecs.find(ssName);
      TEUCHOS_TEST_FOR_EXCEPTION(sideSetMeshSpecIter == sideSetMeshSpecs.end(), std::runtime_error,
          "Cannot find " << ssName << " in sideSetMeshSpecs!\n");
      sideSetMeshSpecIter->second[0]->worksetSize = lateralSidesetWorksetSizeMax;
      sideSetMeshSpecIter->second[0]->singleWorksetSizeAllocation = true;

      // Set lateral workset size to extruded_lateral workset size (special case)
      if (ssName == "extruded_lateralside") {
        sideSetMeshSpecIter = sideSetMeshSpecs.find("lateralside");
        TEUCHOS_TEST_FOR_EXCEPTION(sideSetMeshSpecIter == sideSetMeshSpecs.end(), std::runtime_error,
            "Cannot find lateral in sideSetMeshSpecs!\n");
        sideSetMeshSpecIter->second[0]->worksetSize = lateralSidesetWorksetSizeMax;
        sideSetMeshSpecIter->second[0]->singleWorksetSizeAllocation = true;
      }
    }

    // Set upperside workset size to basalside workset size (special case)
    const std::string ssName = "upperside";
    const auto& sideSetMeshSpecs = this->meshSpecs[0]->sideSetMeshSpecs;
    auto sideSetMeshSpecIter = sideSetMeshSpecs.find(ssName);
    TEUCHOS_TEST_FOR_EXCEPTION(sideSetMeshSpecIter == sideSetMeshSpecs.end(), std::runtime_error,
        "Cannot find " << ssName << " in sideSetMeshSpecs!\n");
    sideSetMeshSpecIter->second[0]->worksetSize = basalWorksetSize;
    sideSetMeshSpecIter->second[0]->singleWorksetSizeAllocation = true;
  }

  // Initialize the requested sideset mesh struct in the mesh
  this->initializeSideSetMeshStructs(comm);
}

Albany::ExtrudedSTKMeshStruct::~ExtrudedSTKMeshStruct()
{
  // Nothing to be done here
}

void Albany::ExtrudedSTKMeshStruct::setFieldData(
    const Teuchos::RCP<const Teuchos_Comm>& comm,
    const AbstractFieldContainer::FieldContainerRequirements& req,
    const Teuchos::RCP<Albany::StateInfoStruct>& sis,
    const unsigned int worksetSize,
    const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis,
    const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& side_set_req)
{
  out->setProcRankAndSize(comm->getRank(), comm->getSize());
  out->setOutputToRootOnly(0);

  // Finish to set up the basal mesh
  Teuchos::RCP<Albany::StateInfoStruct> dummy_sis = Teuchos::rcp(new Albany::StateInfoStruct());
  dummy_sis->createNodalDataBase();
  AbstractFieldContainer::FieldContainerRequirements dummy_req;
  auto it_req = side_set_req.find("basalside");
  auto it_sis = side_set_sis.find("basalside");
  auto& basal_req = (it_req==side_set_req.end() ? dummy_req : it_req->second);
  auto& basal_sis = (it_sis==side_set_sis.end() ? dummy_sis : it_sis->second);

  this->sideSetMeshStructs.at("basalside")->setFieldData (comm, basal_req, basal_sis, worksetSize);

  // Setting up the field container
  this->SetupFieldData(comm, req, sis, worksetSize);

  this->setSideSetFieldData(comm, side_set_req, side_set_sis, worksetSize);
}

void Albany::ExtrudedSTKMeshStruct::setBulkData(
    const Teuchos::RCP<const Teuchos_Comm>& comm,
    const AbstractFieldContainer::FieldContainerRequirements& req,
    const Teuchos::RCP<Albany::StateInfoStruct>& /* sis */,
    const unsigned int worksetSize,
    const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis,
    const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& side_set_req)
{
  // Finish to set up the basal mesh
  Teuchos::RCP<Albany::StateInfoStruct> dummy_sis = Teuchos::rcp(new Albany::StateInfoStruct());
  dummy_sis->createNodalDataBase();
  AbstractFieldContainer::FieldContainerRequirements dummy_req;
  auto it_req = side_set_req.find("basalside");
  auto it_sis = side_set_sis.find("basalside");
  auto& basal_req = (it_req==side_set_req.end() ? dummy_req : it_req->second);
  auto& basal_sis = (it_sis==side_set_sis.end() ? dummy_sis : it_sis->second);

  this->sideSetMeshStructs.at("basalside")->setBulkData (comm, basal_req, basal_sis, worksetSize);

  LayeredMeshOrdering LAYER  = LayeredMeshOrdering::LAYER;
  LayeredMeshOrdering COLUMN = LayeredMeshOrdering::COLUMN;

  bool useGlimmerSpacing = params->get("Use Glimmer Spacing", false);

  stk::mesh::BulkData& bulkData2D = *basalMeshStruct->bulkData;
  stk::mesh::MetaData& metaData2D = *basalMeshStruct->metaData; //bulkData2D.mesh_meta_data();

  std::vector<double> levelsNormalizedThickness(numLayers + 1), temperatureNormalizedZ;

  if(useGlimmerSpacing)
    for (int i = 0; i < numLayers+1; i++)
      levelsNormalizedThickness[numLayers-i] = 1.0- (1.0 - std::pow(double(i) / numLayers + 1.0, -2))/(1.0 - std::pow(2.0, -2));
  else  //uniform layers
    for (int i = 0; i < numLayers+1; i++)
      levelsNormalizedThickness[i] = double(i) / numLayers;

  Teuchos::ArrayRCP<double> layerThicknessRatio(numLayers);
  for (int i = 0; i < numLayers; i++)
    layerThicknessRatio[i] = levelsNormalizedThickness[i+1]-levelsNormalizedThickness[i];

  /*std::cout<< "Levels: ";
  for (int i = 0; i < numLayers+1; i++)
    std::cout<< levelsNormalizedThickness[i] << " ";
  std::cout<< "\n";*/

  stk::mesh::Selector select_owned_in_part = stk::mesh::Selector(metaData2D.universal_part()) & stk::mesh::Selector(metaData2D.locally_owned_part());

  stk::mesh::Selector select_overlap_in_part = stk::mesh::Selector(metaData2D.universal_part()) & (stk::mesh::Selector(metaData2D.locally_owned_part()) | stk::mesh::Selector(metaData2D.globally_shared_part()));

  std::vector<stk::mesh::Entity> cells2D;
  stk::mesh::get_selected_entities(select_overlap_in_part, bulkData2D.buckets(stk::topology::ELEMENT_RANK), cells2D);

  std::vector<stk::mesh::Entity> nodes2D;
  stk::mesh::get_selected_entities(select_overlap_in_part, bulkData2D.buckets(stk::topology::NODE_RANK), nodes2D);

  // For sides, use the 'select_owned_in_part', since boundary sides should belong to just one rank.
  // This for sure eliminate the risk of finding a side with only one element connected, but that is
  // on an 'internal' sideset, which happens to be at the interface of 2 ranks.
  // Note: I'm not 100% sure those internal interface sides would return 1 as the number of connected
  //       elements (they may return 2, storing the ghost), but just in case.
  std::vector<stk::mesh::Entity> sides2D;
  stk::mesh::get_selected_entities(select_owned_in_part, bulkData2D.buckets(metaData2D.side_rank()), sides2D);

  //std::cout << "Num Global Elements: " << maxGlobalElements2D<< " " << globalVerticesStride<< " " << maxGlobalSides2D << std::endl;

  Teuchos::Array<Tpetra_GO> indices(nodes2D.size());
  for (size_t i = 0; i < nodes2D.size(); ++i)
  {
    indices[i] = bulkData2D.identifier(nodes2D[i]) - 1;
  }
  Teuchos::RCP<const Tpetra_Map> nodes_map = Tpetra::createNonContigMapWithNode<LO, Tpetra_GO, KokkosNode>(indices(),comm);

  indices.resize(cells2D.size());
  for (size_t i=0; i<cells2D.size(); ++i)
  {
    indices[i] = bulkData2D.identifier(cells2D[i]) -1;
  }
  Teuchos::RCP<const Tpetra_Map> cells_map = Tpetra::createNonContigMapWithNode<LO, Tpetra_GO, KokkosNode>(indices(),comm);

  indices.resize(sides2D.size());
  for (size_t i=0; i<sides2D.size(); ++i)
  {
    indices[i] = bulkData2D.identifier(sides2D[i]) -1;
  }
  Teuchos::RCP<const Tpetra_Map> sides_map = Tpetra::createNonContigMapWithNode<LO, Tpetra_GO, KokkosNode>(indices(),comm);

  GO globalElemStride = cells_map->getMaxAllGlobalIndex() - cells_map->getMinAllGlobalIndex() + 1;
  GO globalVerticesStride = nodes_map->getMaxAllGlobalIndex() - nodes_map->getMinAllGlobalIndex() + 1;
  GO globalSidesStride    = sides_map->getMaxAllGlobalIndex() - sides_map->getMinAllGlobalIndex() + 1;

  GO elemColumnShift     = (Ordering == COLUMN) ? 1 : globalElemStride;
  int lElemColumnShift   = (Ordering == COLUMN) ? 1 : cells2D.size();
  int elemLayerShift     = (Ordering == LAYER)  ? 1 : numLayers;

  GO vertexColumnShift   = (Ordering == COLUMN) ? 1 : globalVerticesStride;
  int lVertexColumnShift = (Ordering == COLUMN) ? 1 : nodes2D.size();
  int vertexLayerShift   = (Ordering == LAYER)  ? 1 : numLayers + 1;

  GO sideColumnShift     = (Ordering == COLUMN) ? 1 : globalSidesStride;
  int lsideColumnShift   = (Ordering == COLUMN) ? 1 : sides2D.size();
  int sideLayerShift     = (Ordering == LAYER)  ? 1 : numLayers;

  this->layered_mesh_numbering = (Ordering==LAYER) ?
      Teuchos::rcp(new LayeredMeshNumbering<GO>(vertexColumnShift,Ordering,layerThicknessRatio)):
      Teuchos::rcp(new LayeredMeshNumbering<GO>(static_cast<GO>(vertexLayerShift),Ordering,layerThicknessRatio));

  std::vector<double> ltr(layerThicknessRatio.size());
  for(size_t i=0; i< ltr.size(); ++i) {
    ltr[i]=layerThicknessRatio[i];
  }
  fieldContainer->getMeshVectorStates()["layer_thickness_ratio"] = ltr;
  fieldContainer->getMeshScalarIntegerStates()["ordering"] = static_cast<int>(Ordering);
  fieldContainer->getMeshScalarIntegerStates()["stride"] = (Ordering==LAYER) ? lVertexColumnShift : vertexLayerShift;
  fieldContainer->getMeshScalarInteger64States()["global_stride"] = (Ordering==LAYER) ? vertexColumnShift : vertexLayerShift;

  metaData->commit();

  bulkData->modification_begin(); // Begin modifying the mesh

  stk::mesh::PartVector nodePartVec;
  stk::mesh::PartVector singlePartVec(1);
  stk::mesh::PartVector singlePartVecBottom(1);
  stk::mesh::PartVector singlePartVecLateral(1);
  stk::mesh::PartVector singlePartVecTop(1);
  stk::mesh::PartVector emptyPartVec;
  unsigned int ebNo = 0; //element block #???

  singlePartVecBottom[0] = nsPartVec["bottom"];
  singlePartVecTop[0] = nsPartVec["top"];
  singlePartVecLateral[0] = nsPartVec["lateral"];

  typedef AbstractSTKFieldContainer::ScalarFieldType ScalarFieldType;
  typedef AbstractSTKFieldContainer::VectorFieldType VectorFieldType;

  // Fields required for extrusion
  std::string thickness_name = params->get<std::string>("Thickness Field Name","thickness");
  std::string surface_height_name = params->get<std::string>("Surface Height Field Name","surface_height");

  ScalarFieldType* surface_height_field = metaData2D.get_field<ScalarFieldType>(stk::topology::NODE_RANK, surface_height_name);
  ScalarFieldType* thickness_field = metaData2D.get_field<ScalarFieldType>(stk::topology::NODE_RANK, thickness_name);
  AbstractSTKFieldContainer::IntScalarFieldType* proc_rank_field = fieldContainer->getProcRankField();
  VectorFieldType* coordinates_field = fieldContainer->getCoordinatesField();
  stk::mesh::FieldBase const* coordinates_field2d = metaData2D.coordinate_field();

  std::vector<GO> prismMpasIds(NumBaseElemeNodes), prismGlobalIds(2 * NumBaseElemeNodes);

  double *thick_val, *sHeight_val;

  int num_nodes = (numLayers + 1) * nodes2D.size();
  *out << "[ExtrudedSTKMesh] Adding nodes... ";
  out->getOStream()->flush();
  for (int i = 0; i < num_nodes; i++) {
    int ib = (Ordering == LAYER) * (i % lVertexColumnShift) + (Ordering == COLUMN) * (i / vertexLayerShift);
    int il = (Ordering == LAYER) * (i / lVertexColumnShift) + (Ordering == COLUMN) * (i % vertexLayerShift);
    stk::mesh::Entity node;
    stk::mesh::Entity node2d = nodes2D[ib];
    stk::mesh::EntityId node2dId = bulkData2D.identifier(node2d) - 1;
    GO nodeId = il * vertexColumnShift + vertexLayerShift * node2dId + 1;
    if (il == 0)
      node = bulkData->declare_node(nodeId, singlePartVecBottom);
    else if (il == numLayers)
      node = bulkData->declare_node(nodeId, singlePartVecTop);
    else
      node = bulkData->declare_node(nodeId, nodePartVec);

    std::vector<int> sharing_procs;
    bulkData2D.comm_shared_procs( bulkData2D.entity_key(node2d), sharing_procs );
    for(const int proc : sharing_procs) {
      bulkData->add_node_sharing(node, proc);
    }

    double* coord = stk::mesh::field_data(*coordinates_field, node);
    double const* coord2d = (double const*) stk::mesh::field_data(*coordinates_field2d, node2d);
    coord[0] = coord2d[0];
    coord[1] = coord2d[1];

    thick_val = stk::mesh::field_data(*thickness_field, node2d);
    sHeight_val = stk::mesh::field_data(*surface_height_field, node2d);
    coord[2] = sHeight_val[0] - thick_val[0] * (1. - levelsNormalizedThickness[il]);
  }

  *out << "done!\n";
  out->getOStream()->flush();

  GO tetrasLocalIdsOnPrism[3][4];
  singlePartVec[0] = partVec[ebNo];

  *out << "[ExtrudedSTKMesh] Adding elements... ";
  out->getOStream()->flush();
  GO num_cells = cells2D.size() * numLayers;
  for (int i = 0; i < num_cells; i++) {

    int ib = (Ordering == LAYER) * (i % lElemColumnShift) + (Ordering == COLUMN) * (i / elemLayerShift);
    int il = (Ordering == LAYER) * (i / lElemColumnShift) + (Ordering == COLUMN) * (i % elemLayerShift);
    int shift = il * vertexColumnShift;

    //TODO: this could be done only in the first layer and then copied into the other layers
    stk::mesh::Entity const* rel = bulkData2D.begin_nodes(cells2D[ib]);
    for (int j = 0; j < NumBaseElemeNodes; j++) {
      stk::mesh::EntityId node2dId = bulkData2D.identifier(rel[j]) - 1;
      stk::mesh::EntityId mpasLowerId = vertexLayerShift * node2dId;
      stk::mesh::EntityId lowerId = shift + vertexLayerShift * node2dId;
      prismMpasIds[j] = mpasLowerId;
      prismGlobalIds[j] = lowerId;
      prismGlobalIds[j + NumBaseElemeNodes] = lowerId + vertexColumnShift;
    }

    switch (ElemShape)
    {
      case Tetrahedron:
      {
        tetrasFromPrismStructured(&prismMpasIds[0], &prismGlobalIds[0], tetrasLocalIdsOnPrism);

        stk::mesh::EntityId prismId = il * elemColumnShift + elemLayerShift * (bulkData2D.identifier(cells2D[ib]) - 1);
        for (int iTetra = 0; iTetra < 3; iTetra++) {
          stk::mesh::Entity elem = bulkData->declare_element(3 * prismId + iTetra + 1, singlePartVec);
          for (int j = 0; j < 4; j++) {
            stk::mesh::Entity node = bulkData->get_entity(stk::topology::NODE_RANK, tetrasLocalIdsOnPrism[iTetra][j] + 1);
            bulkData->declare_relation(elem, node, j);
          }
          if(proc_rank_field){
            int* p_rank = (int*) stk::mesh::field_data(*proc_rank_field, elem);
            if(p_rank)
              p_rank[0] = comm->getRank();
          }
        }
        break;
      }
      case Wedge:
      case Hexahedron:
      {
        stk::mesh::EntityId prismId = il * elemColumnShift + elemLayerShift * (bulkData2D.identifier(cells2D[ib]) - 1);
        stk::mesh::Entity elem = bulkData->declare_element(prismId + 1, singlePartVec);
        for (int j = 0; j < 2 * NumBaseElemeNodes; j++) {
          stk::mesh::Entity node = bulkData->get_entity(stk::topology::NODE_RANK, prismGlobalIds[j] + 1);
          bulkData->declare_relation(elem, node, j);
        }
        if(proc_rank_field){
          int* p_rank = (int*) stk::mesh::field_data(*proc_rank_field, elem);
          if(p_rank)
            p_rank[0] = comm->getRank();
        }
      }
    }
  }

  *out << "done!\n";
  out->getOStream()->flush();

  int numSubelemOnPrism, numBasalSidePoints;
  int basalSideLID, upperSideLID;

  switch (ElemShape) {
  case Tetrahedron:
    numSubelemOnPrism = 3;
    numBasalSidePoints = 3;
    basalSideLID = 3;  //depends on how the tetrahedron is located in the Prism, see tetraFaceIdOnPrismFaceId below.
    upperSideLID = 1;
    break;
  case Wedge:
    numSubelemOnPrism = 1;
    numBasalSidePoints = 3;
    basalSideLID = 3;  //depends on how the tetrahedron is located in the Prism.
    upperSideLID = 4;
    break;
  case Hexahedron:
    numSubelemOnPrism = 1;
    numBasalSidePoints = 4;
    basalSideLID = 4;  //depends on how the tetrahedron is located in the Prism.
    upperSideLID = 5;
    break;
  }

  // First, we store the lower and upper faces of prisms, which corresponds to triangles of the basal mesh
  // Note: this ensures the side_id on basal sides equals the corresponding cell_id on the basal mesh
  //       which is useful in side_discretization handling.
  // WARNING: these sides do NOT follow column-wise ordering (even if requested).
  singlePartVec[0] = ssPartVec["basalside"];

  *out << "[ExtrudedSTKMesh] Adding basalside sides... ";
  out->getOStream()->flush();
  for (const auto& elem2d : cells2D) {
    stk::mesh::EntityId elem2d_id = bulkData2D.identifier(elem2d) - 1;
    stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), elem2d_id + 1, singlePartVec);
    stk::mesh::Entity elem = bulkData->get_entity(stk::topology::ELEMENT_RANK, elem2d_id * numSubelemOnPrism * elemLayerShift + 1);
    bulkData->declare_relation(elem, side, basalSideLID);

    stk::mesh::Entity const* rel_elemNodes = bulkData->begin_nodes(elem);
    for (int j = 0; j < numBasalSidePoints; j++) {
      stk::mesh::Entity node = rel_elemNodes[this->meshSpecs[0]->ctd.side[basalSideLID].node[j]];
      bulkData->declare_relation(side, node, j);
    }
  }

  *out << "done!\n";
  out->getOStream()->flush();

  singlePartVec[0] = ssPartVec["upperside"];

  GO upperBasalOffset = globalElemStride;

  *out << "[ExtrudedSTKMesh] Adding upperside sides... ";
  out->getOStream()->flush();
  for (const auto& elem2d : cells2D) {
    stk::mesh::EntityId elem2d_id = bulkData2D.identifier(elem2d) - 1;
    stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), elem2d_id + upperBasalOffset + 1, singlePartVec);
    stk::mesh::Entity elem = bulkData->get_entity(stk::topology::ELEMENT_RANK, elem2d_id * numSubelemOnPrism * elemLayerShift + (numLayers - 1) * numSubelemOnPrism * elemColumnShift + 1 + (numSubelemOnPrism - 1));
    bulkData->declare_relation(elem, side, upperSideLID);

    stk::mesh::Entity const* rel_elemNodes = bulkData->begin_nodes(elem);
    for (int j = 0; j < numBasalSidePoints; j++) {
      stk::mesh::Entity node = rel_elemNodes[this->meshSpecs[0]->ctd.side[upperSideLID].node[j]];
      bulkData->declare_relation(side, node, j);
    }
  }

  *out << "done!\n";
  out->getOStream()->flush();

  singlePartVec[0] = ssPartVec["lateralside"];

  //The following two arrays have being computed offline using the computeMap function in .hpp file.

  //tetraFaceIdOnPrismFaceId[ minIndex ][ PrismFaceID ]
  int tetraFaceIdOnPrismFaceId[6][5] = {{0, 1, 2, 3, 1}, {2, 0, 1, 3, 1}, {1, 2, 0, 3, 1}, {2, 1, 0, 1 ,3}, {0, 2, 1, 1, 3}, {1, 0, 2, 1, 3}};

  //tetraAdjacentToPrismLateralFace[ minIndex ][ prismType ][ PrismFaceID ][ iTetra ]. There are to Terahedra adjacent to a Prism face. iTetra in {0,1}
  int tetraAdjacentToPrismLateralFace[6][2][3][2] = { { { { 1, 2 }, { 0, 1 }, { 0, 2 } }, { { 0, 2 }, { 0, 1 }, { 1, 2 } } },
                                                      { { { 0, 2 }, { 1, 2 }, { 0, 1 } }, { { 1, 2 }, { 0, 2 }, { 0, 1 } } },
                                                      { { { 0, 1 }, { 0, 2 }, { 1, 2 } }, { { 0, 1 }, { 1, 2 }, { 0, 2 } } },
                                                      { { { 0, 2 }, { 0, 1 }, { 1, 2 } }, { { 1, 2 }, { 0, 1 }, { 0, 2 } } },
                                                      { { { 1, 2 }, { 0, 2 }, { 0, 1 } }, { { 0, 2 }, { 1, 2 }, { 0, 1 } } },
                                                      { { { 0, 1 }, { 1, 2 }, { 0, 2 } }, { { 0, 1 }, { 0, 2 }, { 1, 2 } } } };

  upperBasalOffset += globalElemStride;

  *out << "[ExtrudedSTKMesh] Adding lateral sides... ";
  out->getOStream()->flush();
  int num_sides = sides2D.size() * numLayers;
  for (int i = 0; i < num_sides; i++) {
    int ib = (Ordering == LAYER) * (i % lsideColumnShift) + (Ordering == COLUMN) * (i / sideLayerShift);
    // if(!isBoundaryside[ib]) continue; //WARNING: assuming that all the edges stored are boundary edges!!

    stk::mesh::Entity side2d = sides2D[ib];

    // We are extracting *all* sides in the basal mesh that are owned by this rank. It could happen that
    // the basal mesh also stores some 'internal' sides. We need to skip those, so check how many elements
    // is this side connected to.
    if (bulkData2D.num_elements(side2d)>1) {
      continue;
    }

    stk::mesh::Entity const* rel = bulkData2D.begin_elements(side2d);
    stk::mesh::ConnectivityOrdinal const* ordinals = bulkData2D.begin_element_ordinals(side2d);

    int il = (Ordering == LAYER) * (i / lsideColumnShift) + (Ordering == COLUMN) * (i % sideLayerShift);
    stk::mesh::Entity elem2d = rel[0];
    stk::mesh::EntityId sideLID = ordinals[0]; //bulkData2D.identifier(rel[0]);

    stk::mesh::EntityId basalElemId = bulkData2D.identifier(elem2d) - 1;
    stk::mesh::EntityId side2dId = bulkData2D.identifier(side2d) - 1;
    switch (ElemShape) {
      case Tetrahedron: {
        rel = bulkData2D.begin_nodes(elem2d);
        for (int j = 0; j < NumBaseElemeNodes; j++) {
          stk::mesh::EntityId node2dId = bulkData2D.identifier(rel[j]) - 1;
          prismMpasIds[j] = vertexLayerShift * node2dId;
        }

        stk::mesh::EntityId sideId = 2 * sideColumnShift * il +  2 * side2dId * sideLayerShift + upperBasalOffset + 1;
        stk::mesh::Entity side0 = bulkData->declare_entity(metaData->side_rank(), sideId, singlePartVec);
        stk::mesh::Entity side1 = bulkData->declare_entity(metaData->side_rank(), sideId + 1, singlePartVec);

        int minIndex;
        int pType = prismType(&prismMpasIds[0], minIndex);
        stk::mesh::EntityId tetraId = 3 * il * elemColumnShift + 3 * elemLayerShift * basalElemId;

        stk::mesh::Entity elem0 = bulkData->get_entity(stk::topology::ELEMENT_RANK, tetraId + tetraAdjacentToPrismLateralFace[minIndex][pType][sideLID][0] + 1);
        stk::mesh::Entity elem1 = bulkData->get_entity(stk::topology::ELEMENT_RANK, tetraId + tetraAdjacentToPrismLateralFace[minIndex][pType][sideLID][1] + 1);

        bulkData->declare_relation(elem0, side0, tetraFaceIdOnPrismFaceId[minIndex][sideLID]);
        bulkData->declare_relation(elem1, side1, tetraFaceIdOnPrismFaceId[minIndex][sideLID]);

        stk::mesh::Entity const* rel_elemNodes0 = bulkData->begin_nodes(elem0);
        stk::mesh::Entity const* rel_elemNodes1 = bulkData->begin_nodes(elem1);
        for (int j = 0; j < 3; j++) {
          stk::mesh::Entity node0 = rel_elemNodes0[this->meshSpecs[0]->ctd.side[tetraFaceIdOnPrismFaceId[minIndex][sideLID]].node[j]];
          bulkData->declare_relation(side0, node0, j);
          bulkData->change_entity_parts(node0, singlePartVecLateral);
          stk::mesh::Entity node1 = rel_elemNodes1[this->meshSpecs[0]->ctd.side[tetraFaceIdOnPrismFaceId[minIndex][sideLID]].node[j]];
          bulkData->declare_relation(side1, node1, j);
          bulkData->change_entity_parts(node1, singlePartVecLateral);
        }
      }

        break;
      case Wedge:
      case Hexahedron: {
        stk::mesh::EntityId sideId = sideColumnShift * il + side2dId * sideLayerShift + upperBasalOffset + 1;
        stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), sideId, singlePartVec);

        stk::mesh::EntityId prismId = il * elemColumnShift + elemLayerShift * basalElemId;
        stk::mesh::Entity elem = bulkData->get_entity(stk::topology::ELEMENT_RANK, prismId + 1);
        bulkData->declare_relation(elem, side, sideLID);

        stk::mesh::Entity const* rel_elemNodes = bulkData->begin_nodes(elem);
        for (int j = 0; j < 4; j++) {
          stk::mesh::Entity node = rel_elemNodes[this->meshSpecs[0]->ctd.side[sideLID].node[j]];
          bulkData->declare_relation(side, node, j);
          bulkData->change_entity_parts(node, singlePartVecLateral);
        }
      }
      break;
    }
  }

  auto partvec = metaData2D.get_mesh_parts();
  for (auto part:partvec) {
    if(part->primary_entity_rank() != metaData2D.side_rank())
      continue;
    stk::mesh::get_selected_entities(stk::mesh::Selector(*part), bulkData2D.buckets(metaData2D.side_rank()), sides2D);
    singlePartVec[0] = ssPartVec["extruded_"+part->name()];
    for (const auto& side2D : sides2D) {
      const stk::mesh::EntityId side2dId = bulkData2D.identifier(side2D) - 1;
      for (int il=0; il<numLayers; ++il) {
        switch (ElemShape) {
          case Tetrahedron: {
            stk::mesh::EntityId sideId = 2 * sideColumnShift * il +  2 * side2dId * sideLayerShift + upperBasalOffset + 1;
            stk::mesh::Entity side0 = bulkData->get_entity(metaData->side_rank(), sideId);
            stk::mesh::Entity side1 = bulkData->get_entity(metaData->side_rank(), sideId + 1);
            bulkData->change_entity_parts(side0, singlePartVec);
            bulkData->change_entity_parts(side1, singlePartVec);
          }
          break;
          case Wedge:
          case Hexahedron: {
            stk::mesh::EntityId sideId = sideColumnShift * il + side2dId * sideLayerShift + upperBasalOffset + 1;
            stk::mesh::Entity side = bulkData->get_entity(metaData->side_rank(), sideId);
            bulkData->change_entity_parts(side, singlePartVec);
          }
          break;
        }
      }
    }
  }

  auto nodepartvec = metaData2D.get_mesh_parts();
  std::vector<stk::mesh::Entity> boundaryNodes2D;
  std::vector<stk::mesh::Entity> nodes;
  for (auto part:nodepartvec) {
    if(part->primary_entity_rank() != stk::topology::NODE_RANK) {
      continue;
    }
    stk::mesh::get_selected_entities(stk::mesh::Selector(*part), bulkData2D.buckets(stk::topology::NODE_RANK), boundaryNodes2D);
    singlePartVecLateral[0] = nsPartVec["extruded_"+part->name()];
    singlePartVecBottom[0] = nsPartVec["basal_"+part->name()];

    for (const auto& node2D : boundaryNodes2D) {
      const stk::mesh::EntityId node2dId = bulkData2D.identifier(node2D) - 1;
      for (int il=0; il<(numLayers+1); ++il) {
        const GO nodeId = il * vertexColumnShift + vertexLayerShift * node2dId + 1;
        stk::mesh::Entity node = bulkData->get_entity(stk::topology::NODE_RANK, nodeId);
        bulkData->change_entity_parts(node, singlePartVecLateral);
      }
      const GO nodeId = vertexLayerShift * node2dId + 1;
      stk::mesh::Entity node = bulkData->get_entity(stk::topology::NODE_RANK, nodeId);
      bulkData->change_entity_parts(node, singlePartVecBottom);
    }
  }

  *out << "done!\n";
  out->getOStream()->flush();

  // Extrude fields
  extrudeBasalFields (nodes2D,cells2D,globalElemStride,globalVerticesStride);
  interpolateBasalLayeredFields (nodes2D,cells2D,levelsNormalizedThickness,globalElemStride,globalVerticesStride);

  // Loading required input fields from file
  this->loadRequiredInputFields (req,comm);

  //Albany::fix_node_sharing(*bulkData);
  bulkData->modification_end();
  fieldAndBulkDataSet = true;

  // Check that the nodeset created from sidesets contain the right number of nodes
  this->checkNodeSetsFromSideSetsIntegrity ();

  // We can finally extract the side set meshes and set the fields and bulk data in all of them
  this->setSideSetBulkData(comm, side_set_req, side_set_sis, worksetSize);

  if (params->get("Export 2D Data",false))
  {
    // We export the basal mesh in GMSH format
    std::ofstream ofile;
    ofile.open (params->get("GMSH 2D Output File Name","basal_mesh.msh"));
    TEUCHOS_TEST_FOR_EXCEPTION (!ofile.is_open(), std::logic_error, "Error! Cannot open file 'basal_mesh.msh'.\n");

    // Preamble
    ofile << "$MeshFormat\n"
          << "2.2 0 8\n"
          << "$EndMeshFormat\n";

    // Nodes
    ofile << "$Nodes\n" << nodes2D.size() << "\n";
    stk::mesh::Entity node2d;
    stk::mesh::EntityId nodeId;
    for (size_t i=0; i<nodes2D.size(); ++i)
    {
      node2d = bulkData2D.get_entity(stk::topology::NODE_RANK, i + 1);
      nodeId = bulkData2D.identifier(nodes2D[i]);

      double const* coord2d = (double const*) stk::mesh::field_data(*coordinates_field2d, node2d);

      ofile << nodeId << " " << coord2d[0] << " " << coord2d[1] << " " << 0. << "\n";
    }
    ofile << "$EndNodes\n";

    // Mesh Elements (including sides)
    ofile << "$Elements\n";
    ofile << sides2D.size()+cells2D.size() << "\n";

    int counter = 1;

    // sides
    for (size_t i=0; i<sides2D.size(); ++i)
    {
      stk::mesh::Entity const* rel = bulkData2D.begin_nodes(sides2D[i]);

      ofile << counter << " " << 1 << " " << 2 << " " << 30 << " " << 1;
      for (int j(0); j<2; ++j)
      {
        stk::mesh::EntityId node2dId = bulkData2D.identifier(rel[j]);
        ofile << " " << node2dId;
      }
      ofile << "\n";
      ++counter;
    }

    // elements
    for (size_t i=0; i<cells2D.size(); ++i)
    {
      stk::mesh::Entity const* rel = bulkData2D.begin_nodes(cells2D[i]);
      ofile << counter << " " << 2 << " " << 2 << " " << 100 << " " << 11;
      for (int j(0); j<3; ++j)
      {
        stk::mesh::EntityId node2dId = bulkData2D.identifier(rel[j]);
        ofile << " " << node2dId;
      }
      ofile << "\n";
      ++counter;
    }
    ofile << "$EndElements\n";

    ofile.close();
  }
}

void Albany::ExtrudedSTKMeshStruct::buildCellSideNodeNumerationMap (const std::string& sideSetName,
                                                                    std::map<GO,GO>& sideMap,
                                                                    std::map<GO,std::vector<int>>& sideNodeMap)
{
  if (sideSetName!="basalside" || Ordering==LayeredMeshOrdering::LAYER || basalMeshStruct->side_maps_present)
  {
    GenericSTKMeshStruct::buildCellSideNodeNumerationMap (sideSetName, sideMap, sideNodeMap);
    return;
  }

  // Extract 2D cells
  stk::mesh::Selector selector = stk::mesh::Selector(basalMeshStruct->metaData->locally_owned_part());
  std::vector<stk::mesh::Entity> cells2D;
  stk::mesh::get_selected_entities(selector, basalMeshStruct->bulkData->buckets(stk::topology::ELEM_RANK), cells2D);

  // If the mesh is already partitioned, this can happen...
  if (cells2D.size()==0)
    return;

  const stk::topology::rank_t SIDE_RANK = metaData->side_rank();
  typedef AbstractSTKFieldContainer::IntScalarFieldType ISFT;
  typedef AbstractSTKFieldContainer::IntVectorFieldType IVFT;
  ISFT* side_to_cell_map   = basalMeshStruct->metaData->get_field<ISFT> (stk::topology::ELEM_RANK, "side_to_cell_map");
  IVFT* side_nodes_ids_map = basalMeshStruct->metaData->get_field<IVFT> (stk::topology::ELEM_RANK, "side_nodes_ids");
  int num_nodes = basalMeshStruct->bulkData->num_nodes(cells2D[0]);
  int* cell3D_id;
  int* side_nodes_ids;
  GO cell2D_GID, side3D_GID;
  const stk::mesh::Entity* cell2D_nodes;
  const stk::mesh::Entity* side3D_nodes;
  const stk::mesh::Entity* side_cells;
  std::vector<stk::mesh::EntityId> cell2D_nodes_ids(num_nodes), side3D_nodes_ids(num_nodes);
  std::vector<stk::mesh::EntityId> original_nodes_ids(num_nodes);
  for (const auto& cell2D : cells2D)
  {
    // Get the stk field data
    cell3D_id      = stk::mesh::field_data(*side_to_cell_map, cell2D);
    side_nodes_ids = stk::mesh::field_data(*side_nodes_ids_map, cell2D);

    // The side-id is assumed equal to the cell-id in the side mesh...
    side3D_GID = cell2D_GID = basalMeshStruct->bulkData->identifier(cell2D)-1;
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
    cell2D_nodes = basalMeshStruct->bulkData->begin_nodes(cell2D);
    side3D_nodes = bulkData->begin_nodes(side3D);
    for (int i(0); i<num_nodes; ++i)
    {
      cell2D_nodes_ids[i] = basalMeshStruct->bulkData->identifier(cell2D_nodes[i]);
      side3D_nodes_ids[i] = bulkData->identifier(side3D_nodes[i]);
      // Need to recover the "original" (basal) node id that generated the extrude mesh
      // node Id; it's the id the node would have if the mesh had been built with LAYER ordering..
      original_nodes_ids[i] = ((bulkData->identifier(side3D_nodes[i]) -1) / (numLayers + 1)) + 1;
    }

    for (int i(0); i<num_nodes; ++i)
    {
      auto it = std::find(cell2D_nodes_ids.begin(),cell2D_nodes_ids.end(),original_nodes_ids[i]);
      sideNodeMap[side3D_GID][i] = std::distance(cell2D_nodes_ids.begin(),it);
      side_nodes_ids[std::distance(cell2D_nodes_ids.begin(),it)] = side3D_nodes_ids[i];
    }
  }

  // Just in case this method gets called twice
  basalMeshStruct->side_maps_present = true;
}

void Albany::ExtrudedSTKMeshStruct::interpolateBasalLayeredFields (const std::vector<stk::mesh::Entity>& nodes2d,
                                                                   const std::vector<stk::mesh::Entity>& cells2d,
                                                                   const std::vector<double>& levelsNormalizedThickness,
                                                                   GO maxGlobalCells2dId, GO maxGlobalNodes2dId)
{
  Teuchos::Array<std::string> node_fields_names, cell_fields_names;
  Teuchos::Array<int> node_fields_ranks, cell_fields_ranks;
  if (params->isParameter("Interpolate Basal Node Layered Fields"))
  {
    TEUCHOS_TEST_FOR_EXCEPTION (!params->isParameter("Basal Node Layered Fields Ranks"), std::logic_error,
                                "Error! To interpolate basal node layered fields, you also need the 'Basal Node Layered Fields Ranks' parameter.\n");
    node_fields_names = params->get<Teuchos::Array<std::string> >("Interpolate Basal Node Layered Fields");
    node_fields_ranks = params->get<Teuchos::Array<int> >("Basal Node Layered Fields Ranks");
  }

  if (params->isParameter("Interpolate Basal Elem Layered Fields"))
  {
    TEUCHOS_TEST_FOR_EXCEPTION (!params->isParameter("Basal Elem Layered Fields Ranks"), std::logic_error,
                                "Error! To interpolate basal elem layered fields, you also need the 'Basal Elem Layered Fields Ranks' parameter.\n");
    cell_fields_names = params->get<Teuchos::Array<std::string> >("Interpolate Basal Elem Layered Fields");
    cell_fields_ranks = params->get<Teuchos::Array<int> >("Basal Elem Layered Fields Ranks");
  }

  if (node_fields_names.size()==0 && cell_fields_names.size()==0)
    return;

  *out << "[ExtrudedSTKMesh] Interpolating layered basal fields...\n";
  out->getOStream()->flush();

  // NOTE: a scalar layered field is stored as a vector field,
  //       a vector layered field is stored as a tensor field.

  // Typedefs
  typedef Albany::AbstractSTKFieldContainer::ScalarFieldType SFT;
  typedef Albany::AbstractSTKFieldContainer::VectorFieldType VFT;
  typedef Albany::AbstractSTKFieldContainer::TensorFieldType TFT;

  // Utility constants
  const int numNodes2d = nodes2d.size();
  const int numCells2d = cells2d.size();

  stk::mesh::BulkData& bulkData2d = *basalMeshStruct->bulkData;
  stk::mesh::MetaData& metaData2d = *basalMeshStruct->metaData;

  int numNodeFields = node_fields_names.size();
  int numCellFields = cell_fields_names.size();
  int numScalars;
  double *values2d, *values3d;

  int il0,il1;
  double h0;

  LayeredMeshOrdering COLUMN = LayeredMeshOrdering::COLUMN;

  std::string ranks[4] = {"ERROR!","Scalar","Vector","Tensor"};
  std::vector<double> fieldLayersCoords;

  // Interpolate node fields
  for (int ifield=0; ifield<numNodeFields; ++ifield)
  {
    *out << "  - Interpolating " << ranks[node_fields_ranks[ifield]] << " field " << node_fields_names[ifield] << "...";
    out->getOStream()->flush();

    // We also need to load the normalized layers coordinates
    std::string tmp_str = node_fields_names[ifield] + "_NLC";
    auto it = basalMeshStruct->getFieldContainer()->getMeshVectorStates().find(tmp_str);
    TEUCHOS_TEST_FOR_EXCEPTION (it==basalMeshStruct->getFieldContainer()->getMeshVectorStates().end(), std::logic_error,
                                "Error in ExtrudedSTKMeshStruct: normalized layers coords for layered field '" <<
                                node_fields_names[ifield] << "' not found.\n");

    fieldLayersCoords = it->second;
    int numFieldLayers = fieldLayersCoords.size();

    for (int inode=0; inode<numNodes2d; ++inode)
    {
      const stk::mesh::Entity& node2d = nodes2d[inode];
      stk::mesh::EntityId node2dId = bulkData2d.identifier(node2d) - 1;

      // Extracting 2d data only once
      switch (node_fields_ranks[ifield])
      {
        case 1:
        {
          VFT* field2d = metaData2d.get_field<VFT>(stk::topology::NODE_RANK, node_fields_names[ifield]);
          TEUCHOS_TEST_FOR_EXCEPTION (field2d==0, std::runtime_error, "Error! Cannot interpolate layered field '" << node_fields_names[ifield] << "' since it is not present in the 2d mesh.\n");
          values2d = stk::mesh::field_data(*field2d,node2d);
          break;
        }
        case 2:
        {
          TFT* field2d = metaData2d.get_field<TFT>(stk::topology::NODE_RANK, node_fields_names[ifield]);
          TEUCHOS_TEST_FOR_EXCEPTION (field2d==0, std::runtime_error, "Error! Cannot interpolate layered field '" << node_fields_names[ifield] << "' since it is not present in the 2d mesh.\n");
          numScalars = stk::mesh::field_scalars_per_entity(*field2d,node2d);
          values2d = stk::mesh::field_data(*field2d,node2d);
          break;
        }
        default:
          TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Invalid/unsupported field rank.\n");
      }

      // Loop on the layers
      for (int il=0; il<=numLayers; ++il)
      {
        // Retrieve 3D node
        int node3dId = Ordering==COLUMN ? node2dId*(numLayers+1) + il + 1 : il*maxGlobalNodes2dId + node2dId + 1;
        stk::mesh::Entity node3d = bulkData->get_entity(stk::topology::NODE_RANK, node3dId);

        // Find where the mesh layer stands in the field layers
        double meshLayerCoord = levelsNormalizedThickness[il];

        auto where = std::upper_bound(fieldLayersCoords.begin(),fieldLayersCoords.end(),meshLayerCoord);
        il1 = std::distance(fieldLayersCoords.begin(),where);
        if (il1==0) // mesh layer is below the first field layer
        {
          il0 = 0;
          h0 = 0.; // Useless, (the 2 values in the convex combination will be the same) but for clarity we fix it to 0
        }
        else if (il1==numFieldLayers) // mesh layer is above the last field layer
        {
          il0 = il1 = numFieldLayers-1;
          h0 = 0.; // Useless, (the 2 values in the convex combination will be the same) but for clarity we fix it to 0
        }
        else
        {
          il0 = il1-1;
          h0 = (fieldLayersCoords[il1] - meshLayerCoord) / (fieldLayersCoords[il1] - fieldLayersCoords[il0]);
        }

        // Extracting 3d pointer and stuffing the right data in it
        // TODO: find a way for ExtrudedSTKMeshStruct to automatically add the fields to be interpolated, so the user does not have to
        //       specify them twice (in the 2d mesh and in the 3d mesh) in the input file. Note: this must be done before you call
        //       the SetupFieldData method, which adds all the fields to the stk mesh.
        switch (node_fields_ranks[ifield])
        {
          case 1:
          {
            SFT* field3d = metaData->get_field<SFT> (stk::topology::NODE_RANK, node_fields_names[ifield]);
            TEUCHOS_TEST_FOR_EXCEPTION (field3d==0, std::runtime_error, "Error! Cannot interpolate layered field '" << node_fields_names[ifield] << "' since it is not present in the 3d mesh. Perhaps you forgot to specify it in the section 'Required Fields Info' of the extruded mesh?\n");
            values3d = stk::mesh::field_data(*field3d,node3d);
            values3d[0] = h0*values2d[il0]+(1-h0)*values2d[il1];
            break;
          }
          case 2:
          {
            VFT* field3d = metaData->get_field<VFT> (stk::topology::NODE_RANK, node_fields_names[ifield]);
            TEUCHOS_TEST_FOR_EXCEPTION (field3d==0, std::runtime_error, "Error! Cannot interpolate layered field '" << node_fields_names[ifield] << "' since it is not present in the 3d mesh. Perhaps you forgot to specify it in the section 'Required Fields Info' of the extruded mesh?\n");
            values3d = stk::mesh::field_data(*field3d,node3d);
            for (int j=0; j<numScalars/numFieldLayers; ++j)
              values3d[j] = h0*values2d[j*numFieldLayers+il0]+(1-h0)*values2d[j*numFieldLayers+il1];
            break;
          }
          default:
            TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Invalid/unsupported field rank.\n");
        }
      }
    }
    *out << "done!\n";
  }

  // Extrude cell fields
  for (int ifield=0; ifield<numCellFields; ++ifield)
  {
    *out << "  - Interpolating " << ranks[cell_fields_ranks[ifield]] << " field " << cell_fields_names[ifield] << "...";
    // We also need to load the normalized layers coordinates
    std::string tmp_str = cell_fields_names[ifield] + "_NLC";
    auto it = basalMeshStruct->getFieldContainer()->getMeshVectorStates().find(tmp_str);
    TEUCHOS_TEST_FOR_EXCEPTION (it==basalMeshStruct->getFieldContainer()->getMeshVectorStates().end(), std::logic_error,
                                "Error in ExtrudedSTKMeshStruct: normalized layers coords for layered field '" <<
                                cell_fields_names[ifield] << "' not found.\n");
    fieldLayersCoords = it->second;

    int numFieldLayers = fieldLayersCoords.size();

    for (int icell=0; icell<numCells2d; ++icell)
    {
      const stk::mesh::Entity& cell2d = cells2d[icell];
      stk::mesh::EntityId cell2dId = bulkData2d.identifier(cell2d) - 1;

      // Extracting the 2d data only once
      switch (cell_fields_ranks[ifield])
      {
        case 1:
        {
          VFT* field2d = metaData2d.get_field<VFT>(stk::topology::ELEMENT_RANK, cell_fields_names[ifield]);
          TEUCHOS_TEST_FOR_EXCEPTION (field2d==0, std::runtime_error, "Error! Cannot interpolate layered field '" << cell_fields_names[ifield] << "' since it is not present in the 2d mesh.\n");
          values2d = stk::mesh::field_data(*field2d,cell2d);
          break;
        }
        case 2:
        {
          TFT* field2d = metaData2d.get_field<TFT>(stk::topology::ELEMENT_RANK, cell_fields_names[ifield]);
          TEUCHOS_TEST_FOR_EXCEPTION (field2d==0, std::runtime_error, "Error! Cannot interpolate layered field '" << cell_fields_names[ifield] << "' since it is not present in the 2d mesh.\n");
          numScalars = stk::mesh::field_scalars_per_entity(*field2d,cell2d);
          values2d = stk::mesh::field_data(*field2d,cell2d);
          break;
        }
        default:
          TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Invalid/unsupported field rank.\n");
      }

      // Loop on the layers
      for (int il=0; il<numLayers; ++il)
      {
        // Retrieving the id of the 3d cells
        stk::mesh::EntityId prismId = Ordering==COLUMN ? numLayers*cell2dId + il : maxGlobalCells2dId*il + cell2dId;
        std::vector<stk::mesh::Entity> cells3d;
        switch (ElemShape)
        {
          case Tetrahedron:
            cells3d.push_back (bulkData->get_entity(stk::topology::ELEMENT_RANK, 3*prismId+1));
            cells3d.push_back (bulkData->get_entity(stk::topology::ELEMENT_RANK, 3*prismId+2));
            cells3d.push_back (bulkData->get_entity(stk::topology::ELEMENT_RANK, 3*prismId+3));
            break;
          case Wedge:
          case Hexahedron:
            cells3d.push_back (bulkData->get_entity(stk::topology::ELEMENT_RANK, prismId+1));
        }

        // Since the
        double meshLayerCoord = 0.5*(levelsNormalizedThickness[il] + levelsNormalizedThickness[il+1]);

        // Find where the mesh layer stands in the field layers
        auto where = std::upper_bound(fieldLayersCoords.begin(),fieldLayersCoords.end(),meshLayerCoord);
        il1 = std::distance(fieldLayersCoords.begin(),where);
        if (il1==0) // mesh layer is below the first field layer
        {
          il0 = 0;
          h0 = 0.; // Useless, (the 2 values in the convex combination will be the same) but for clarity we fix it to 0
        }
        else if (il1==numFieldLayers) // mesh layer is above the last field layer
        {
          il0 = il1 = numFieldLayers-1;
          h0 = 0.; // Useless, (the 2 values in the convex combination will be the same) but for clarity we fix it to 0
        }
        else
        {
          il0 = il1-1;
          h0 = (fieldLayersCoords[il1] - meshLayerCoord) / (fieldLayersCoords[il1] - fieldLayersCoords[il0]);
        }

        // Extracting 3d pointer and stuffing the right data in it
        // TODO: find a way for ExtrudedSTKMeshStruct to automatically add the fields to be interpolated, so the user does not have to
        //       specify them twice (in the 2d mesh and in the 3d mesh) in the input file. Note: this must be done before you call
        //       the SetupFieldData method, which adds all the fields to the stk mesh.
        switch (cell_fields_ranks[ifield])
        {
          case 1:
          {
            SFT* field3d = metaData->get_field<SFT> (stk::topology::ELEMENT_RANK, cell_fields_names[ifield]);
            TEUCHOS_TEST_FOR_EXCEPTION (field3d==0, std::runtime_error, "Error! Cannot interpolate layered field '" << cell_fields_names[ifield] << "' since it is not present in the 3d mesh. Perhaps you forgot to specify it in the section 'Required Fields Info' of the extruded mesh?\n");
            for (auto& cell3d : cells3d)
            {
              values3d = stk::mesh::field_data(*field3d,cell3d);
              values3d[0] = h0*values2d[il0]+(1-h0)*values2d[il1];
            }
            break;
          }
          case 2:
          {
            VFT* field3d = metaData->get_field<VFT> (stk::topology::ELEMENT_RANK, cell_fields_names[ifield]);
            TEUCHOS_TEST_FOR_EXCEPTION (field3d==0, std::runtime_error, "Error! Cannot interpolate layered field '" << cell_fields_names[ifield] << "' since it is not present in the 3d mesh. Perhaps you forgot to specify it in the section 'Required Fields Info' of the extruded mesh?\n");
            for (auto& cell3d : cells3d)
            {
              values3d = stk::mesh::field_data(*field3d,cell3d);
              for (int j=0; j<numScalars; ++j)
                values3d[j] = h0*values2d[j*numFieldLayers+il0]+(1-h0)*values2d[j*numFieldLayers+il1];
            }
            break;
          }
          default:
            TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Invalid/unsupported field rank.\n");
        }
      }
    }
    *out << "done!\n";
  }
}

void Albany::ExtrudedSTKMeshStruct::extrudeBasalFields (const std::vector<stk::mesh::Entity>& nodes2d,
                                                        const std::vector<stk::mesh::Entity>& cells2d,
                                                        GO maxGlobalCells2dId, GO maxGlobalNodes2dId)
{
  Teuchos::Array<std::string> node_fields_names, cell_fields_names;
  Teuchos::Array<int> node_fields_ranks, cell_fields_ranks;
  if (params->isParameter("Extrude Basal Node Fields"))
  {
    TEUCHOS_TEST_FOR_EXCEPTION (!params->isParameter("Basal Node Fields Ranks"), std::logic_error,
                                "Error! To extrude basal node fields, you also need the 'Basal Node Fields Ranks' parameter.\n");
    node_fields_names = params->get<Teuchos::Array<std::string> >("Extrude Basal Node Fields");
    node_fields_ranks = params->get<Teuchos::Array<int> >("Basal Node Fields Ranks");
  }

  if (params->isParameter("Extrude Basal Elem Fields"))
  {
    TEUCHOS_TEST_FOR_EXCEPTION (!params->isParameter("Basal Elem Fields Ranks"), std::logic_error,
                                "Error! To extrude basal elem fields, you also need the 'Basal Elem Fields Ranks' parameter.\n");
    cell_fields_names = params->get<Teuchos::Array<std::string> >("Extrude Basal Elem Fields");
    cell_fields_ranks = params->get<Teuchos::Array<int> >("Basal Elem Fields Ranks");
  }

  if (node_fields_names.size()==0 && cell_fields_names.size()==0)
    return;

  *out << "[ExtrudedSTKMesh] Extruding basal fields...\n";
  out->getOStream()->flush();

  // Typedefs
  typedef Albany::AbstractSTKFieldContainer::ScalarFieldType SFT;
  typedef Albany::AbstractSTKFieldContainer::VectorFieldType VFT;
  typedef Albany::AbstractSTKFieldContainer::TensorFieldType TFT;

  // Utility constants
  const int numNodes2d = nodes2d.size();
  const int numCells2d = cells2d.size();

  stk::mesh::BulkData& bulkData2d = *basalMeshStruct->bulkData;
  stk::mesh::MetaData& metaData2d = *basalMeshStruct->metaData;

  int numNodeFields = node_fields_names.size();
  int numCellFields = cell_fields_names.size();
  int numScalars;
  double *values2d, *values3d;

  LayeredMeshOrdering COLUMN = LayeredMeshOrdering::COLUMN;

  // Extrude node fields
  for (int ifield=0; ifield<numNodeFields; ++ifield)
  {
    switch (node_fields_ranks[ifield])
    {
      case 1:
      {
        *out << "  - Extruding Scalar Node field " << node_fields_names[ifield] << "...";
        out->getOStream()->flush();

        SFT* field2d = metaData2d.get_field<SFT>(stk::topology::NODE_RANK, node_fields_names[ifield]);
        SFT* field3d = metaData->get_field<SFT> (stk::topology::NODE_RANK, node_fields_names[ifield]);

        // TODO: find a way for ExtrudedSTKMeshStruct to automatically add the fields to be extruded, so the user does not have to
        //       specify them twice (in the 2d mesh and in the 3d mesh) in the input file. Note: this must be done before you call
        //       the SetupFieldData method, which adds all the fields to the stk mesh.
        TEUCHOS_TEST_FOR_EXCEPTION (field2d==0, std::runtime_error, "Error! Cannot extrude field '" << node_fields_names[ifield] << "' since it is not present in the 2d mesh.\n");
        TEUCHOS_TEST_FOR_EXCEPTION (field3d==0, std::runtime_error, "Error! Cannot extrude field '" << node_fields_names[ifield] << "' since it is not present in the 3d mesh. Perhaps you forgot to specify it in the section 'Required Fields Info' of the extruded mesh?\n");

        for (int inode=0; inode<numNodes2d; ++inode)
        {
          const stk::mesh::Entity& node2d = nodes2d[inode];
          stk::mesh::EntityId node2dId = bulkData2d.identifier(node2d) - 1;
          values2d = stk::mesh::field_data(*field2d,node2d);
          for (int il=0; il<=numLayers; ++il)
          {
            // Retrieve 3D node
            int node3dId = Ordering==COLUMN ? node2dId*(numLayers+1) + il + 1 : il*maxGlobalNodes2dId + node2dId + 1;
            stk::mesh::Entity node3d = bulkData->get_entity(stk::topology::NODE_RANK, node3dId);

            values3d = stk::mesh::field_data(*field3d,node3d);
            values3d[0] = values2d[0];
          }
        }
        break;
      }
      case 2:
      {
        *out << "  - Extruding Vector Node field " << node_fields_names[ifield] << "...";
        out->getOStream()->flush();

        VFT* field2d = metaData2d.get_field<VFT>(stk::topology::NODE_RANK, node_fields_names[ifield]);
        VFT* field3d = metaData->get_field<VFT> (stk::topology::NODE_RANK, node_fields_names[ifield]);

        // TODO: find a way for ExtrudedSTKMeshStruct to automatically add the fields to be extruded, so the user does not have to
        //       specify them twice (in the 2d mesh and in the 3d mesh) in the input file. Note: this must be done before you call
        //       the SetupFieldData method, which adds all the fields to the stk mesh.
        TEUCHOS_TEST_FOR_EXCEPTION (field2d==0, std::runtime_error, "Error! Cannot extrude field '" << node_fields_names[ifield] << "' since it is not present in the 2d mesh.\n");
        TEUCHOS_TEST_FOR_EXCEPTION (field3d==0, std::runtime_error, "Error! Cannot extrude field '" << node_fields_names[ifield] << "' since it is not present in the 3d mesh. Perhaps you forgot to specify it in the section 'Required Fields Info' of the extruded mesh?\n");

        for (int inode=0; inode<numNodes2d; ++inode)
        {
          const stk::mesh::Entity& node2d = nodes2d[inode];
          stk::mesh::EntityId node2dId = bulkData2d.identifier(node2d) - 1;
          values2d = stk::mesh::field_data(*field2d,node2d);
          numScalars = stk::mesh::field_scalars_per_entity(*field2d,node2d);
          for (int il=0; il<=numLayers; ++il)
          {
            // Retrieve 3D node
            int node3dId = Ordering==COLUMN ? node2dId*(numLayers+1)+il+1 : il*maxGlobalNodes2dId+node2dId + 1;
            stk::mesh::Entity node3d = bulkData->get_entity(stk::topology::NODE_RANK, node3dId);

            values3d = stk::mesh::field_data(*field3d,node3d);
            for (int j=0; j<numScalars; ++j)
              values3d[j] = values2d[j];
          }
        }
        break;
      }
      case 3:
      {
        *out << "  - Extruding Tensor Node field " << node_fields_names[ifield] << "...";
        out->getOStream()->flush();

        TFT* field2d = metaData2d.get_field<TFT>(stk::topology::NODE_RANK, node_fields_names[ifield]);
        TFT* field3d = metaData->get_field<TFT> (stk::topology::NODE_RANK, node_fields_names[ifield]);

        // TODO: find a way for ExtrudedSTKMeshStruct to automatically add the fields to be extruded, so the user does not have to
        //       specify them twice (in the 2d mesh and in the 3d mesh) in the input file. Note: this must be done before you call
        //       the SetupFieldData method, which adds all the fields to the stk mesh.
        TEUCHOS_TEST_FOR_EXCEPTION (field2d==0, std::runtime_error, "Error! Cannot extrude field '" << node_fields_names[ifield] << "' since it is not present in the 2d mesh.\n");
        TEUCHOS_TEST_FOR_EXCEPTION (field3d==0, std::runtime_error, "Error! Cannot extrude field '" << node_fields_names[ifield] << "' since it is not present in the 3d mesh. Perhaps you forgot to specify it in the section 'Required Fields Info' of the extruded mesh?\n");

        for (int inode=0; inode<numNodes2d; ++inode)
        {
          const stk::mesh::Entity& node2d = nodes2d[inode];
          stk::mesh::EntityId node2dId = bulkData2d.identifier(node2d) - 1;
          values2d = stk::mesh::field_data(*field2d,node2d);
          numScalars = stk::mesh::field_scalars_per_entity(*field2d,node2d);
          for (int il=0; il<=numLayers; ++il)
          {
            // Retrieve 3D node
            int node3dId = Ordering==COLUMN ? node2dId*(numLayers+1)+il+1 : il*maxGlobalNodes2dId+node2dId + 1;
            stk::mesh::Entity node3d = bulkData->get_entity(stk::topology::NODE_RANK, node3dId);

            values3d = stk::mesh::field_data(*field3d,node3d);
            for (int j=0; j<numScalars; ++j)
              values3d[j] = values2d[j];
          }
        }
        break;
      }
      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Invalid/unsupported field rank.\n");
    }
    *out << "done!\n";
  }

  // Extrude cell fields
  for (int ifield=0; ifield<numCellFields; ++ifield)
  {
    switch (cell_fields_ranks[ifield])
    {
      case 1:
      {
        *out << "  - Extruding Scalar Elem field " << cell_fields_names[ifield] << "...";
        out->getOStream()->flush();

        SFT* field2d = metaData2d.get_field<SFT>(stk::topology::ELEMENT_RANK, cell_fields_names[ifield]);
        SFT* field3d = metaData->get_field<SFT> (stk::topology::ELEMENT_RANK, cell_fields_names[ifield]);

        // TODO: find a way for ExtrudedSTKMeshStruct to automatically add the fields to be extruded, so the user does not have to
        //       specify them twice (in the 2d mesh and in the 3d mesh) in the input file. Note: this must be done before you call
        //       the SetupFieldData method, which adds all the fields to the stk mesh.
        TEUCHOS_TEST_FOR_EXCEPTION (field2d==0, std::runtime_error, "Error! Cannot extrude field '" << cell_fields_names[ifield] << "' since it is not present in the 2d mesh.\n");
        TEUCHOS_TEST_FOR_EXCEPTION (field3d==0, std::runtime_error, "Error! Cannot extrude field '" << cell_fields_names[ifield] << "' since it is not present in the 3d mesh. Perhaps you forgot to specify it in the section 'Required Fields Info' of the extruded mesh?\n");

        for (int icell=0; icell<numCells2d; ++icell)
        {
          const stk::mesh::Entity& cell2d = cells2d[icell];
          stk::mesh::EntityId cell2dId = bulkData2d.identifier(cell2d) - 1;
          values2d = stk::mesh::field_data(*field2d,cell2d);
          for (int il=0; il<numLayers; ++il)
          {
            // Retrieving the id of the 3d cells
            stk::mesh::EntityId prismId = Ordering==COLUMN ? numLayers*cell2dId + il : maxGlobalCells2dId*il + cell2dId;
            std::vector<stk::mesh::Entity> cells3d;
            switch (ElemShape)
            {
              case Tetrahedron:
                cells3d.push_back (bulkData->get_entity(stk::topology::ELEMENT_RANK, 3*prismId+1));
                cells3d.push_back (bulkData->get_entity(stk::topology::ELEMENT_RANK, 3*prismId+2));
                cells3d.push_back (bulkData->get_entity(stk::topology::ELEMENT_RANK, 3*prismId+3));
                break;
              case Wedge:
              case Hexahedron:
                cells3d.push_back (bulkData->get_entity(stk::topology::ELEMENT_RANK, prismId+1));
            }

            // Stuffing the 3d fields
            for (auto& cell3d : cells3d)
            {
              values3d = stk::mesh::field_data(*field3d,cell3d);
              values3d[0] = values2d[0];
            }
          }
        }
        break;
      }
      case 2:
      {
        *out << "  - Extruding Vector Elem field " << cell_fields_names[ifield] << "...";
        out->getOStream()->flush();

        VFT* field2d = metaData2d.get_field<VFT>(stk::topology::ELEMENT_RANK, cell_fields_names[ifield]);
        VFT* field3d = metaData->get_field<VFT> (stk::topology::ELEMENT_RANK, cell_fields_names[ifield]);

        // TODO: find a way for ExtrudedSTKMeshStruct to automatically add the fields to be extruded, so the user does not have to
        //       specify them twice (in the 2d mesh and in the 3d mesh) in the input file. Note: this must be done before you call
        //       the SetupFieldData method, which adds all the fields to the stk mesh.
        TEUCHOS_TEST_FOR_EXCEPTION (field2d==0, std::runtime_error, "Error! Cannot extrude field '" << cell_fields_names[ifield] << "' since it is not present in the 2d mesh.\n");
        TEUCHOS_TEST_FOR_EXCEPTION (field3d==0, std::runtime_error, "Error! Cannot extrude field '" << cell_fields_names[ifield] << "' since it is not present in the 3d mesh. Perhaps you forgot to specify it in the section 'Required Fields Info' of the extruded mesh?\n");

        for (int icell=0; icell<numCells2d; ++icell)
        {
          const stk::mesh::Entity& cell2d = cells2d[icell];
          stk::mesh::EntityId cell2dId = bulkData2d.identifier(cell2d) - 1;
          numScalars = stk::mesh::field_scalars_per_entity(*field2d,cell2d);
          values2d = stk::mesh::field_data(*field2d,cell2d);
          for (int il=0; il<numLayers; ++il)
          {
            // Retrieving the id of the 3d cells
            stk::mesh::EntityId prismId = Ordering==COLUMN ? numLayers*cell2dId + il + 1 : maxGlobalCells2dId*il + cell2dId + 1;
            std::vector<stk::mesh::Entity> cells3d;
            switch (ElemShape)
            {
              case Tetrahedron:
                cells3d.push_back (bulkData->get_entity(stk::topology::ELEMENT_RANK, prismId+2));
                cells3d.push_back (bulkData->get_entity(stk::topology::ELEMENT_RANK, prismId+1));
              case Wedge:
              case Hexahedron:
                cells3d.push_back (bulkData->get_entity(stk::topology::ELEMENT_RANK, prismId));
            }

            // Stuffing the 3d fields
            for (auto& cell3d : cells3d)
            {
              values3d = stk::mesh::field_data(*field3d,cell3d);
              values3d[0] = values2d[0];

              for (int j=0; j<numScalars; ++j)
                values3d[j] = values2d[j];
            }
          }
        }
        break;
      }
      case 3:
      {
        *out << "  - Extruding Tensor Elem field " << cell_fields_names[ifield] << "...";
        out->getOStream()->flush();

        TFT* field2d = metaData2d.get_field<TFT>(stk::topology::ELEMENT_RANK, cell_fields_names[ifield]);
        TFT* field3d = metaData->get_field<TFT> (stk::topology::ELEMENT_RANK, cell_fields_names[ifield]);

        // TODO: find a way for ExtrudedSTKMeshStruct to automatically add the fields to be extruded, so the user does not have to
        //       specify them twice (in the 2d mesh and in the 3d mesh) in the input file. Note: this must be done before you call
        //       the SetupFieldData method, which adds all the fields to the stk mesh.
        TEUCHOS_TEST_FOR_EXCEPTION (field2d==0, std::runtime_error, "Error! Cannot extrude field '" << cell_fields_names[ifield] << "' since it is not present in the 2d mesh.\n");
        TEUCHOS_TEST_FOR_EXCEPTION (field3d==0, std::runtime_error, "Error! Cannot extrude field '" << cell_fields_names[ifield] << "' since it is not present in the 3d mesh. Perhaps you forgot to specify it in the section 'Required Fields Info' of the extruded mesh?\n");

        for (int icell=0; icell<numCells2d; ++icell)
        {
          const stk::mesh::Entity& cell2d = cells2d[icell];
          stk::mesh::EntityId cell2dId = bulkData2d.identifier(cell2d) - 1;
          numScalars = stk::mesh::field_scalars_per_entity(*field2d,cell2d);
          values2d = stk::mesh::field_data(*field2d,cell2d);
          for (int il=0; il<numLayers; ++il)
          {
            // Retrieving the id of the 3d cells
            stk::mesh::EntityId prismId = Ordering==COLUMN ? numLayers*cell2dId + il + 1 : maxGlobalCells2dId*il + cell2dId + 1;
            std::vector<stk::mesh::Entity> cells3d;
            switch (ElemShape)
            {
              case Tetrahedron:
                cells3d.push_back (bulkData->get_entity(stk::topology::ELEMENT_RANK, prismId+2));
                cells3d.push_back (bulkData->get_entity(stk::topology::ELEMENT_RANK, prismId+1));
              case Wedge:
              case Hexahedron:
                cells3d.push_back (bulkData->get_entity(stk::topology::ELEMENT_RANK, prismId));
            }

            // Stuffing the 3d fields
            for (auto& cell3d : cells3d)
            {
              values3d = stk::mesh::field_data(*field3d,cell3d);
              values3d[0] = values2d[0];

              for (int j=0; j<numScalars; ++j)
                values3d[j] = values2d[j];
            }
          }
          break;
        }
      }
      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Invalid/unsupported field rank.\n");
    }
    *out << "done!\n";
  }
}

Teuchos::RCP<const Teuchos::ParameterList> Albany::ExtrudedSTKMeshStruct::getValidDiscretizationParameters() const {
  Teuchos::RCP<Teuchos::ParameterList> validPL = this->getValidGenericSTKParameters("Valid Extruded_DiscParams");
  validPL->set<bool>("Export 2D Data", "", "If true, exports the 2D mesh in GMSH format");
  validPL->set<Teuchos::Array<std::string> >("Extrude Basal Node Fields", Teuchos::Array<std::string>(), "List of basal node fields to be extruded");
  validPL->set<Teuchos::Array<std::string> >("Extrude Basal Elem Fields", Teuchos::Array<std::string>(), "List of basal elem fields to be extruded");
  validPL->set<Teuchos::Array<int> >("Basal Node Fields Ranks", Teuchos::Array<int>(), "Ranks of basal node fields to be extruded");
  validPL->set<Teuchos::Array<int> >("Basal Elem Fields Ranks", Teuchos::Array<int>(), "Ranks of basal elem fields to be extruded");
  validPL->set<Teuchos::Array<std::string> >("Interpolate Basal Node Layered Fields", Teuchos::Array<std::string>(), "List of basal node layered fields to be interpolated");
  validPL->set<Teuchos::Array<std::string> >("Interpolate Basal Elem Layered Fields", Teuchos::Array<std::string>(), "List of basal node layered fields to be interpolated");
  validPL->set<Teuchos::Array<int> >("Basal Node Layered Fields Ranks", Teuchos::Array<int>(), "List of basal node layered fields to be interpolated");
  validPL->set<Teuchos::Array<int> >("Basal Elem Layered Fields Ranks", Teuchos::Array<int>(), "List of basal node layered fields to be interpolated");
  validPL->set<std::string>("GMSH 2D Output File Name", "", "File Name for GMSH 2D Basal Mesh Export");
  validPL->set<std::string>("Exodus Input File Name", "", "File Name For Exodus Mesh Input");
  validPL->set<std::string>("Element Shape", "Hexahedron", "Shape of the Element: Tetrahedron, Wedge, Hexahedron");
  validPL->set<int>("NumLayers", 10, "Number of vertical Layers of the extruded mesh. In a vertical column, the mesh will have numLayers+1 nodes");
  validPL->set<bool>("Use Glimmer Spacing", false, "When true, the layer spacing is computed according to Glimmer formula (layers are denser close to the bedrock)");
  validPL->set<bool>("Columnwise Ordering", false, "True for Columnwise ordering, false for Layerwise ordering");

  validPL->set<std::string>("Thickness Field Name","thickness","Name of the 'thickness' field to use for extrusion");
  validPL->set<std::string>("Surface Height Field Name","surface_height","Name of the 'surface_height' field to use for extrusion");

  return validPL;
}
