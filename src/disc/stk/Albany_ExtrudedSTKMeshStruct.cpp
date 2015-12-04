//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>

#include "Albany_ExtrudedSTKMeshStruct.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_IossSTKMeshStruct.hpp"
#include "Albany_TmplSTKMeshStruct.hpp"
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

//#include <stk_mesh/fem/FEMHelpers.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "Albany_Utils.hpp"

const Tpetra::global_size_t INVALID = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid ();

Albany::ExtrudedSTKMeshStruct::ExtrudedSTKMeshStruct(const Teuchos::RCP<Teuchos::ParameterList>& params,
                                                     const Teuchos::RCP<const Teuchos_Comm>& comm) :
    GenericSTKMeshStruct(params, Teuchos::null, 3), out(Teuchos::VerboseObjectBase::getDefaultOStream()), periodic(false)
{
  params->validateParameters(*getValidDiscretizationParameters(), 0);

  std::string ebn = "Element Block 0";
  partVec[0] = &metaData->declare_part(ebn, stk::topology::ELEMENT_RANK);
  ebNameToIndex[ebn] = 0;

#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*partVec[0]);
#endif

  std::vector<std::string> nsNames;
  std::string nsn = "lateral";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = &metaData->declare_part(nsn, stk::topology::NODE_RANK);
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn = "internal";
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

  std::vector<std::string> ssNames;
  std::string ssnLat = "lateralside";
  std::string ssnBottom = "basalside";
  std::string ssnTop = "upperside";

  ssNames.push_back(ssnLat);
  ssNames.push_back(ssnBottom);
  ssNames.push_back(ssnTop);
  ssPartVec[ssnLat] = &metaData->declare_part(ssnLat, metaData->side_rank());
  ssPartVec[ssnBottom] = &metaData->declare_part(ssnBottom, metaData->side_rank());
  ssPartVec[ssnTop] = &metaData->declare_part(ssnTop, metaData->side_rank());
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*ssPartVec[ssnLat]);
  stk::io::put_io_part_attribute(*ssPartVec[ssnBottom]);
  stk::io::put_io_part_attribute(*ssPartVec[ssnTop]);
#endif

  Teuchos::RCP<Teuchos::ParameterList> params2D;
  if (params->isSublist("Side Set Discretizations"))
  {
    params2D = Teuchos::rcp(new Teuchos::ParameterList(params->sublist("Side Set Discretizations").sublist("basalside")));
  }
  else
  {
    // Old style: the 2D parameter are mixed with the 3D
    params2D = Teuchos::rcp(new Teuchos::ParameterList());
    params2D->set("Use Serial Mesh", params->get("Use Serial Mesh", false));
    params2D->set("Exodus Input File Name", params->get("Exodus Input File Name", "IceSheet.exo"));
  }

  if (params2D->isParameter("Method"))
  {
    std::string method = params2D->get<std::string>("Method");

    if (method=="STK2D")
    {
      basalMeshStruct = Teuchos::rcp(new Albany::TmplSTKMeshStruct<2>(params2D, adaptParams, comm));
    }
    else if (method=="Ioss")
    {
      basalMeshStruct = Teuchos::rcp(new Albany::IossSTKMeshStruct(params2D, adaptParams, comm));
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameterValue, "Error in ExtrudedSTKMeshStruct: only 'STK2D' and 'Ioss' accepted for Method in basal mesh\n");
    }
  }
  else
  {
    // If no method is provided, we assume it is Ioss (backward compatibility)
    basalMeshStruct = Teuchos::rcp(new Albany::IossSTKMeshStruct(params2D, adaptParams, comm));
  }
  sideSetMeshStructs["basalside"] = basalMeshStruct;

  std::string shape = params->get("Element Shape", "Hexahedron");
  std::string basalside_name;
  if(shape == "Tetrahedron")  {
    ElemShape = Tetrahedron;
    basalside_name = shards::getCellTopologyData<shards::Triangle<3> >()->name;
  }
  else if (shape == "Wedge")  {
    ElemShape = Wedge;
    basalside_name = shards::getCellTopologyData<shards::Triangle<3> >()->name;
  }
  else if (shape == "Hexahedron") {
    ElemShape = Hexahedron;
    basalside_name = shards::getCellTopologyData<shards::Quadrilateral<4> >()->name;
  }
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameterValue,
              std::endl << "Error in ExtrudedSTKMeshStruct: Element Shape " << shape << " not recognized. Possible values: Tetrahedron, Wedge, Hexahedron");

  std::string elem2d_name(basalMeshStruct->getMeshSpecs()[0]->ctd.base->name);
  TEUCHOS_TEST_FOR_EXCEPTION(basalside_name != elem2d_name, Teuchos::Exceptions::InvalidParameterValue,
                std::endl << "Error in ExtrudedSTKMeshStruct: Expecting topology name of elements of 2d mesh to be " <<  basalside_name << " but it is " << elem2d_name);


  switch (ElemShape) {
  case Tetrahedron:
    stk::mesh::set_cell_topology<shards::Tetrahedron<4> >(*partVec[0]);
    stk::mesh::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnBottom]);
    stk::mesh::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnTop]);
    stk::mesh::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnLat]);
    NumBaseElemeNodes = 3;
    break;
  case Wedge:
    stk::mesh::set_cell_topology<shards::Wedge<6> >(*partVec[0]);
    stk::mesh::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnBottom]);
    stk::mesh::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnTop]);
    stk::mesh::set_cell_topology<shards::Quadrilateral<4> >(*ssPartVec[ssnLat]);
    NumBaseElemeNodes = 3;
    break;
  case Hexahedron:
    stk::mesh::set_cell_topology<shards::Hexahedron<8> >(*partVec[0]);
    stk::mesh::set_cell_topology<shards::Quadrilateral<4> >(*ssPartVec[ssnBottom]);
    stk::mesh::set_cell_topology<shards::Quadrilateral<4> >(*ssPartVec[ssnTop]);
    stk::mesh::set_cell_topology<shards::Quadrilateral<4> >(*ssPartVec[ssnLat]);
    NumBaseElemeNodes = 4;
    break;
  }

  numDim = 3;
  int cub = params->get("Cubature Degree", 3);
  int basalWorksetSize = basalMeshStruct->getMeshSpecs()[0]->worksetSize;
  int worksetSizeMax = params->get("Workset Size", 50);
  int worksetSize = this->computeWorksetSize(worksetSizeMax, basalWorksetSize);

  const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();

  this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub, nsNames, ssNames, worksetSize, partVec[0]->name(), ebNameToIndex, this->interleavedOrdering));

  // Initialize the (possible) other side set meshes
  this->initializeSideSetMeshStructs(comm);
}

Albany::ExtrudedSTKMeshStruct::~ExtrudedSTKMeshStruct()
{
  // Nothing to be done here
}

void Albany::ExtrudedSTKMeshStruct::setFieldAndBulkData(
    const Teuchos::RCP<const Teuchos_Comm>& comm,
    const Teuchos::RCP<Teuchos::ParameterList>& params,
    const unsigned int neq_,
    const AbstractFieldContainer::FieldContainerRequirements& req,
    const Teuchos::RCP<Albany::StateInfoStruct>& sis,
    const unsigned int worksetSize,
    const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis,
    const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& side_set_req)
{
  // Finish to set up the basal mesh
  Teuchos::RCP<Teuchos::ParameterList> params2D;
  if (params->isSublist("Side Set Discretizations"))
  {
    params2D = Teuchos::rcp(new Teuchos::ParameterList(params->sublist("Side Set Discretizations").sublist("basalside")));
  }
  else
  {
    // Old style: the 2D parameter are mixed with the 3D
    params2D = Teuchos::rcp(new Teuchos::ParameterList());
    params2D->set("Use Serial Mesh", params->get("Use Serial Mesh", false));
    params2D->set("Exodus Input File Name", params->get("Exodus Input File Name", "IceSheet.exo"));
  }
  Teuchos::RCP<Albany::StateInfoStruct> dummy_sis = Teuchos::rcp(new Albany::StateInfoStruct());
  dummy_sis->createNodalDataBase();
  AbstractFieldContainer::FieldContainerRequirements dummy_req;
  auto it_req = side_set_req.find("basalside");
  auto it_sis = side_set_sis.find("basalside");
  auto& basal_req = (it_req==side_set_req.end() ? dummy_req : it_req->second);
  auto& basal_sis = (it_sis==side_set_sis.end() ? dummy_sis : it_sis->second);
  this->sideSetMeshStructs.at("basalside")->setFieldAndBulkData (comm, params2D, neq_, basal_req, basal_sis, worksetSize);

  // Setting up the field container
  this->SetupFieldData(comm, neq_, req, sis, worksetSize);

  int numLayers = params->get("NumLayers", 10);
  bool useGlimmerSpacing = params->get("Use Glimmer Spacing", false);
  GO maxGlobalElements2D = 0;
  GO maxGlobalVertices2dId = 0;
  GO numGlobalVertices2D = 0;
  GO maxGlobalEdges2D = 0;
  bool Ordering = params->get("Columnwise Ordering", false);
  bool isTetra = true;

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

  stk::mesh::Selector select_edges = stk::mesh::Selector(*metaData2D.get_part("LateralSide")) & (stk::mesh::Selector(metaData2D.locally_owned_part()) | stk::mesh::Selector(metaData2D.globally_shared_part()));

  std::vector<stk::mesh::Entity> cells2D;
  stk::mesh::get_selected_entities(select_overlap_in_part, bulkData2D.buckets(stk::topology::ELEMENT_RANK), cells2D);

  std::vector<stk::mesh::Entity> nodes2D;
  stk::mesh::get_selected_entities(select_overlap_in_part, bulkData2D.buckets(stk::topology::NODE_RANK), nodes2D);

  std::vector<stk::mesh::Entity> edges2D;
  stk::mesh::get_selected_entities(select_edges, bulkData2D.buckets(metaData2D.side_rank()), edges2D);

  GO maxOwnedElements2D(0), maxOwnedNodes2D(0), maxOwnedSides2D(0), numOwnedNodes2D(0);
  for (int i = 0; i < cells2D.size(); i++)
    maxOwnedElements2D = std::max(maxOwnedElements2D, (GO) bulkData2D.identifier(cells2D[i]));
  for (int i = 0; i < nodes2D.size(); i++)
    maxOwnedNodes2D = std::max(maxOwnedNodes2D, (GO) bulkData2D.identifier(nodes2D[i]));
  for (int i = 0; i < edges2D.size(); i++)
    maxOwnedSides2D = std::max(maxOwnedSides2D, (GO) bulkData2D.identifier(edges2D[i]));
  numOwnedNodes2D = stk::mesh::count_selected_entities(select_owned_in_part, bulkData2D.buckets(stk::topology::NODE_RANK));


  //WARNING Currently GO == long int. For gcc compiler, long == long long, however this might not be true with other compilers.

  //comm->MaxAll(&maxOwnedElements2D, &maxGlobalElements2D, 1);
  Teuchos::reduceAll<int, GO>(*comm, Teuchos::REDUCE_MAX, maxOwnedElements2D, Teuchos::ptr(&maxGlobalElements2D));
  //comm->MaxAll(&maxOwnedNodes2D, &maxGlobalVertices2dId, 1);
  Teuchos::reduceAll<int, GO>(*comm, Teuchos::REDUCE_MAX, maxOwnedNodes2D, Teuchos::ptr(&maxGlobalVertices2dId));
  //comm->MaxAll(&maxOwnedSides2D, &maxGlobalEdges2D, 1);
  Teuchos::reduceAll<int, GO>(*comm, Teuchos::REDUCE_MAX, maxOwnedSides2D, Teuchos::ptr(&maxGlobalEdges2D));
  //comm->SumAll(&numOwnedNodes2D, &numGlobalVertices2D, 1);
  //The following should not be int int...
  Teuchos::reduceAll<int, GO>(*comm, Teuchos::REDUCE_SUM, 1, &numOwnedNodes2D, &numGlobalVertices2D);

  if (comm->getRank() == 0) std::cout << "Importing ascii files ...";

  //std::cout << "Num Global Elements: " << maxGlobalElements2D<< " " << maxGlobalVertices2dId<< " " << maxGlobalEdges2D << std::endl;

  Teuchos::Array<GO> indices(nodes2D.size());
  for (int i = 0; i < nodes2D.size(); ++i)
    indices[i] = bulkData2D.identifier(nodes2D[i]) - 1;

  Teuchos::RCP<const Tpetra_Map>
    nodes_map = Tpetra::createNonContigMapWithNode<LO, GO> (
      indices(), comm, KokkosClassic::Details::getNode<KokkosNode>());
  int numMyElements = (comm->getRank() == 0) ? numGlobalVertices2D : 0;
  Teuchos::RCP<const Tpetra_Map> serial_nodes_map = Teuchos::rcp(new const Tpetra_Map(INVALID, numMyElements, 0, comm));
  Teuchos::RCP<Tpetra_Import> importOperator = Teuchos::rcp(new Tpetra_Import(serial_nodes_map, nodes_map));

  Teuchos::RCP<Tpetra_Vector> sHeightVec;
  Teuchos::RCP<Tpetra_Vector> thickVec;
  Teuchos::RCP<Tpetra_Vector> bTopographyVec;
  Teuchos::RCP<Tpetra_Vector> bFrictionVec;
  Teuchos::RCP<Tpetra_MultiVector> temperatureVecInterp;
  Teuchos::RCP<Tpetra_MultiVector> sVelocityVec;
  Teuchos::RCP<Tpetra_MultiVector> velocityRMSVec;

  if (comm->getRank() == 0)
    std::cout << " done." << std::endl;

  GO elemColumnShift = (Ordering == 1) ? 1 : maxGlobalElements2D;
  int lElemColumnShift = (Ordering == 1) ? 1 : cells2D.size();
  int elemLayerShift = (Ordering == 0) ? 1 : numLayers;

  GO vertexColumnShift = (Ordering == 1) ? 1 : maxGlobalVertices2dId;
  int lVertexColumnShift = (Ordering == 1) ? 1 : nodes2D.size();
  int vertexLayerShift = (Ordering == 0) ? 1 : numLayers + 1;

  GO edgeColumnShift = (Ordering == 1) ? 1 : maxGlobalEdges2D;
  int lEdgeColumnShift = (Ordering == 1) ? 1 : edges2D.size();
  int edgeLayerShift = (Ordering == 0) ? 1 : numLayers;

  this->layered_mesh_numbering = (Ordering==0) ?
      Teuchos::rcp(new LayeredMeshNumbering<LO>(lVertexColumnShift,Ordering,layerThicknessRatio)):
      Teuchos::rcp(new LayeredMeshNumbering<LO>(vertexLayerShift,Ordering,layerThicknessRatio));

  metaData->commit();

  bulkData->modification_begin(); // Begin modifying the mesh

  stk::mesh::PartVector nodePartVec;
  stk::mesh::PartVector singlePartVec(1);
  stk::mesh::PartVector emptyPartVec;
  unsigned int ebNo = 0; //element block #???

  singlePartVec[0] = nsPartVec["bottom"];

  typedef AbstractSTKFieldContainer::ScalarFieldType ScalarFieldType;
  typedef AbstractSTKFieldContainer::VectorFieldType VectorFieldType;
  typedef AbstractSTKFieldContainer::QPScalarFieldType ElemScalarFieldType;

  // Fields required for extrusion
  ScalarFieldType* surface_height_field = metaData2D.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "surface_height");
  ScalarFieldType* thickness_field = metaData2D.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "thickness");
  AbstractSTKFieldContainer::IntScalarFieldType* proc_rank_field = fieldContainer->getProcRankField();
  VectorFieldType* coordinates_field = fieldContainer->getCoordinatesField();
  stk::mesh::FieldBase const* coordinates_field2d = metaData2D.coordinate_field();

  std::vector<GO> prismMpasIds(NumBaseElemeNodes), prismGlobalIds(2 * NumBaseElemeNodes);

  double *thick_val, *sHeight_val;
  for (int i = 0; i < (numLayers + 1) * nodes2D.size(); i++) {
    int ib = (Ordering == 0) * (i % lVertexColumnShift) + (Ordering == 1) * (i / vertexLayerShift);
    int il = (Ordering == 0) * (i / lVertexColumnShift) + (Ordering == 1) * (i % vertexLayerShift);
    stk::mesh::Entity node;
    stk::mesh::Entity node2d = nodes2D[ib];
    stk::mesh::EntityId node2dId = bulkData2D.identifier(node2d) - 1;
    GO nodeId = il * vertexColumnShift + vertexLayerShift * node2dId + 1;
    if (il == 0)
      node = bulkData->declare_entity(stk::topology::NODE_RANK, nodeId, singlePartVec);
    else
      node = bulkData->declare_entity(stk::topology::NODE_RANK, nodeId, nodePartVec);

    std::vector<int> sharing_procs;
    bulkData2D.comm_shared_procs( bulkData2D.entity_key(node2d), sharing_procs );
    for(int iproc=0; iproc<sharing_procs.size(); ++iproc)
      bulkData->add_node_sharing(node, sharing_procs[iproc]);

    double* coord = stk::mesh::field_data(*coordinates_field, node);
    double const* coord2d = (double const*) stk::mesh::field_data(*coordinates_field2d, node2d);
    coord[0] = coord2d[0];
    coord[1] = coord2d[1];

    thick_val = stk::mesh::field_data(*thickness_field, node2d);
    sHeight_val = stk::mesh::field_data(*surface_height_field, node2d);
    coord[2] = sHeight_val[0] - thick_val[0] * (1. - levelsNormalizedThickness[il]);
  }

  GO tetrasLocalIdsOnPrism[3][4];
  singlePartVec[0] = partVec[ebNo];
  for (int i = 0; i < cells2D.size() * numLayers; i++) {

    int ib = (Ordering == 0) * (i % lElemColumnShift) + (Ordering == 1) * (i / elemLayerShift);
    int il = (Ordering == 0) * (i / lElemColumnShift) + (Ordering == 1) * (i % elemLayerShift);
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
          stk::mesh::Entity elem = bulkData->declare_entity(stk::topology::ELEMENT_RANK, 3 * prismId + iTetra + 1, singlePartVec);
          for (int j = 0; j < 4; j++) {
            stk::mesh::Entity node = bulkData->get_entity(stk::topology::NODE_RANK, tetrasLocalIdsOnPrism[iTetra][j] + 1);
            bulkData->declare_relation(elem, node, j);
          }
          int* p_rank = (int*) stk::mesh::field_data(*proc_rank_field, elem);
          p_rank[0] = comm->getRank();
        }
        break;
      }
      case Wedge:
      case Hexahedron:
      {
        stk::mesh::EntityId prismId = il * elemColumnShift + elemLayerShift * (bulkData2D.identifier(cells2D[ib]) - 1);
        stk::mesh::Entity elem = bulkData->declare_entity(stk::topology::ELEMENT_RANK, prismId + 1, singlePartVec);
        for (int j = 0; j < 2 * NumBaseElemeNodes; j++) {
          stk::mesh::Entity node = bulkData->get_entity(stk::topology::NODE_RANK, prismGlobalIds[j] + 1);
          bulkData->declare_relation(elem, node, j);
        }
        int* p_rank = (int*) stk::mesh::field_data(*proc_rank_field, elem);
        p_rank[0] = comm->getRank();
      }
    }
  }

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

  for (int i = 0; i < cells2D.size(); i++) {
    stk::mesh::Entity elem2d = cells2D[i];
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

  singlePartVec[0] = ssPartVec["upperside"];

  GO upperBasalOffset = maxGlobalElements2D;

  for (int i = 0; i < cells2D.size(); i++) {
    stk::mesh::Entity elem2d = cells2D[i];
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

  upperBasalOffset += maxGlobalElements2D;
  for (int i = 0; i < edges2D.size() * numLayers; i++) {
    int ib = (Ordering == 0) * (i % lEdgeColumnShift) + (Ordering == 1) * (i / edgeLayerShift);
    // if(!isBoundaryEdge[ib]) continue; //WARNING: assuming that all the edges stored are boundary edges!!

    stk::mesh::Entity edge2d = edges2D[ib];
    stk::mesh::Entity const* rel = bulkData2D.begin_elements(edge2d);
    stk::mesh::ConnectivityOrdinal const* ordinals = bulkData2D.begin_element_ordinals(edge2d);

    int il = (Ordering == 0) * (i / lEdgeColumnShift) + (Ordering == 1) * (i % edgeLayerShift);
    stk::mesh::Entity elem2d = rel[0];
    stk::mesh::EntityId edgeLID = ordinals[0]; //bulkData2D.identifier(rel[0]);

    stk::mesh::EntityId basalElemId = bulkData2D.identifier(elem2d) - 1;
    stk::mesh::EntityId Edge2dId = bulkData2D.identifier(edge2d) - 1;
    switch (ElemShape) {
    case Tetrahedron: {
      rel = bulkData2D.begin_nodes(elem2d);
      for (int j = 0; j < NumBaseElemeNodes; j++) {
        stk::mesh::EntityId node2dId = bulkData2D.identifier(rel[j]) - 1;
        prismMpasIds[j] = vertexLayerShift * node2dId;
      }
      int minIndex;
      int pType = prismType(&prismMpasIds[0], minIndex);
      stk::mesh::EntityId tetraId = 3 * il * elemColumnShift + 3 * elemLayerShift * basalElemId;

      stk::mesh::Entity elem0 = bulkData->get_entity(stk::topology::ELEMENT_RANK, tetraId + tetraAdjacentToPrismLateralFace[minIndex][pType][edgeLID][0] + 1);
      stk::mesh::Entity elem1 = bulkData->get_entity(stk::topology::ELEMENT_RANK, tetraId + tetraAdjacentToPrismLateralFace[minIndex][pType][edgeLID][1] + 1);

      stk::mesh::Entity side0 = bulkData->declare_entity(metaData->side_rank(), 2 * edgeColumnShift * il +  2 * Edge2dId * edgeLayerShift + upperBasalOffset + 1, singlePartVec);
      stk::mesh::Entity side1 = bulkData->declare_entity(metaData->side_rank(), 2 * edgeColumnShift * il +  2 * Edge2dId * edgeLayerShift + upperBasalOffset + 1 + 1, singlePartVec);

      bulkData->declare_relation(elem0, side0, tetraFaceIdOnPrismFaceId[minIndex][edgeLID]);
      bulkData->declare_relation(elem1, side1, tetraFaceIdOnPrismFaceId[minIndex][edgeLID]);

      stk::mesh::Entity const* rel_elemNodes0 = bulkData->begin_nodes(elem0);
      stk::mesh::Entity const* rel_elemNodes1 = bulkData->begin_nodes(elem1);
      for (int j = 0; j < 3; j++) {
     //   std::cout << j <<", " << edgeLID << ", " << minIndex << ", " << tetraFaceIdOnPrismFaceId[minIndex][edgeLID] << ","  << std::endl;
        stk::mesh::Entity node0 = rel_elemNodes0[this->meshSpecs[0]->ctd.side[tetraFaceIdOnPrismFaceId[minIndex][edgeLID]].node[j]];
        bulkData->declare_relation(side0, node0, j);
        stk::mesh::Entity node1 = rel_elemNodes1[this->meshSpecs[0]->ctd.side[tetraFaceIdOnPrismFaceId[minIndex][edgeLID]].node[j]];
        bulkData->declare_relation(side1, node1, j);
      }
    }

      break;
    case Wedge:
    case Hexahedron: {
      stk::mesh::EntityId prismId = il * elemColumnShift + elemLayerShift * basalElemId;
      stk::mesh::Entity elem = bulkData->get_entity(stk::topology::ELEMENT_RANK, prismId + 1);
      stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), edgeColumnShift * il +Edge2dId * edgeLayerShift + upperBasalOffset + 1, singlePartVec);
      bulkData->declare_relation(elem, side, edgeLID);

      stk::mesh::Entity const* rel_elemNodes = bulkData->begin_nodes(elem);
      for (int j = 0; j < 4; j++) {
        stk::mesh::Entity node = rel_elemNodes[this->meshSpecs[0]->ctd.side[edgeLID].node[j]];
        bulkData->declare_relation(side, node, j);
      }
    }
    break;
    }
  }

  // Extrude fields
  extrudeBasalFields (nodes2D,cells2D,maxGlobalElements2D,maxGlobalVertices2dId);
  interpolateBasalLayeredFields (nodes2D,cells2D,levelsNormalizedThickness,maxGlobalElements2D,maxGlobalVertices2dId);

  //Albany::fix_node_sharing(*bulkData);
  bulkData->modification_end();
  fieldAndBulkDataSet = true;

  // We can finally extract the side set meshes and set the fields and bulk data in all of them
  this->finalizeSideSetMeshStructs(comm, side_set_req, side_set_sis, worksetSize);

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
    for (int i(0); i<nodes2D.size(); ++i)
    {
      node2d = bulkData2D.get_entity(stk::topology::NODE_RANK, i + 1);
      nodeId = bulkData2D.identifier(nodes2D[i]);

      double const* coord2d = (double const*) stk::mesh::field_data(*coordinates_field2d, node2d);

      ofile << nodeId << " " << coord2d[0] << " " << coord2d[1] << " " << 0. << "\n";
    }
    ofile << "$EndNodes\n";

    // Mesh Elements (including edges)
    ofile << "$Elements\n";
    ofile << edges2D.size()+cells2D.size() << "\n";

    int counter = 1;

    // edges
    for (int i(0); i<edges2D.size(); ++i)
    {
      stk::mesh::Entity const* rel = bulkData2D.begin_nodes(edges2D[i]);

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
    for (int i(0); i<cells2D.size(); ++i)
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

void Albany::ExtrudedSTKMeshStruct::interpolateBasalLayeredFields (const std::vector<stk::mesh::Entity>& nodes2d,
                                                                   const std::vector<stk::mesh::Entity>& cells2d,
                                                                   const std::vector<double>& levelsNormalizedThickness,
                                                                   GO maxGlobalCells2dId, GO maxGlobalNodes2dId)
{
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

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

  // NOTE: a scalar layered field is stored as a vector field,
  //       a vector layered field is stored as a tensor field.

  // Typedefs
  typedef Albany::AbstractSTKFieldContainer::ScalarFieldType SFT;
  typedef Albany::AbstractSTKFieldContainer::VectorFieldType VFT;
  typedef Albany::AbstractSTKFieldContainer::TensorFieldType TFT;

  // Utility constants
  const int numLayers  = params->get("NumLayers", 10);
  const int numNodes2d = nodes2d.size();
  const int numCells2d = cells2d.size();
  const bool ColOrdering  = params->get("Columnwise Ordering", false);

  stk::mesh::BulkData& bulkData2d = *basalMeshStruct->bulkData;
  stk::mesh::MetaData& metaData2d = *basalMeshStruct->metaData;

  int numNodeFields = node_fields_names.size();
  int numCellFields = cell_fields_names.size();
  int numScalars;
  double *values2d, *values3d;

  int il0,il1;
  double h0;

  std::string ranks[4] = {"ERROR!","Scalar","Vector","Tensor"};
  std::vector<double> fieldLayersCoords;
  // Extrude node fields
  for (int ifield=0; ifield<numNodeFields; ++ifield)
  {
    *out << "Interpolating " << ranks[node_fields_ranks[ifield]] << " field " << node_fields_names[ifield] << " from basal mesh\n";

    // We also need to load the normalized layers coordinates
    auto it = basalMeshStruct->getFieldContainer()->getMeshVectorStates().find(node_fields_names[ifield]);
    TEUCHOS_TEST_FOR_EXCEPTION (it!=basalMeshStruct->getFieldContainer()->getMeshVectorStates().end(), std::logic_error,
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
          values2d = stk::mesh::field_data(*field2d,node2d);
          break;
        }
        case 2:
        {
          TFT* field2d = metaData2d.get_field<TFT>(stk::topology::NODE_RANK, node_fields_names[ifield]);
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
        int node3dId = ColOrdering ? node2dId*(numLayers+1) + il + 1 : il*maxGlobalNodes2dId + node2dId + 1;
        stk::mesh::Entity node3d = bulkData->get_entity(stk::topology::NODE_RANK, node3dId);

        // Find where the mesh layer stands in the field layers
        double meshLayerCoord = levelsNormalizedThickness[il];

        auto where = std::upper_bound(fieldLayersCoords.begin(),fieldLayersCoords.end(),meshLayerCoord);
        il1 = std::distance(where,fieldLayersCoords.begin());
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

        // Extracting 3d pointer and stuffing the right data in them
        switch (node_fields_ranks[ifield])
        {
          case 1:
          {
            SFT* field3d = metaData->get_field<SFT> (stk::topology::NODE_RANK, node_fields_names[ifield]);
            values3d = stk::mesh::field_data(*field3d,node3d);
            values3d[0] = h0*values2d[il0]+(1-h0)*values2d[il1];
            break;
          }
          case 2:
          {
            VFT* field3d = metaData->get_field<VFT> (stk::topology::NODE_RANK, node_fields_names[ifield]);
            values3d = stk::mesh::field_data(*field3d,node3d);
            for (int j=0; j<numScalars; ++j)
              values3d[j] = h0*values2d[j*numFieldLayers+il0]+(1-h0)*values2d[j*numFieldLayers+il1];
            break;
          }
          default:
            TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Invalid/unsupported field rank.\n");
        }
      }
    }
  }

  // Extrude cell fields
  for (int ifield=0; ifield<numCellFields; ++ifield)
  {
    *out << "Interpolating " << ranks[cell_fields_ranks[ifield]] << " field " << cell_fields_names[ifield] << " from basal mesh\n";
    // We also need to load the normalized layers coordinates
    auto it = basalMeshStruct->getFieldContainer()->getMeshVectorStates().find(cell_fields_names[ifield]);
    TEUCHOS_TEST_FOR_EXCEPTION (it!=basalMeshStruct->getFieldContainer()->getMeshVectorStates().end(), std::logic_error,
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
          values2d = stk::mesh::field_data(*field2d,cell2d);
          break;
        }
        case 2:
        {
          TFT* field2d = metaData2d.get_field<TFT>(stk::topology::ELEMENT_RANK, cell_fields_names[ifield]);
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
        stk::mesh::EntityId prismId = ColOrdering ? numLayers*cell2dId + il + 1 : maxGlobalCells2dId*il + cell2dId + 1;
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

        // Since the
        double meshLayerCoord = 0.5*(levelsNormalizedThickness[il] + levelsNormalizedThickness[il+1]);

        // Find where the mesh layer stands in the field layers
        auto where = std::upper_bound(fieldLayersCoords.begin(),fieldLayersCoords.end(),meshLayerCoord);
        il1 = std::distance(where,fieldLayersCoords.begin());
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

        for (auto& cell3d : cells3d)
        {
          switch (cell_fields_ranks[ifield])
          {
            case 1:
            {
              SFT* field3d = metaData->get_field<SFT> (stk::topology::ELEMENT_RANK, cell_fields_names[ifield]);
              values3d = stk::mesh::field_data(*field3d,cell3d);
              values3d[0] = h0*values2d[il0]+(1-h0)*values2d[il1];
              break;
            }
            case 2:
            {
              VFT* field3d = metaData->get_field<VFT> (stk::topology::ELEMENT_RANK, cell_fields_names[ifield]);
              values3d = stk::mesh::field_data(*field3d,cell3d);
              for (int j=0; j<numScalars; ++j)
                values3d[j] = h0*values2d[j*numFieldLayers+il0]+(1-h0)*values2d[j*numFieldLayers+il1];
              break;
            }
            default:
              TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Invalid/unsupported field rank.\n");
          }
        }
      }
    }
  }
}

void Albany::ExtrudedSTKMeshStruct::extrudeBasalFields (const std::vector<stk::mesh::Entity>& nodes2d,
                                                        const std::vector<stk::mesh::Entity>& cells2d,
                                                        GO maxGlobalCells2dId, GO maxGlobalNodes2dId)
{
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

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

  // Typedefs
  typedef Albany::AbstractSTKFieldContainer::ScalarFieldType SFT;
  typedef Albany::AbstractSTKFieldContainer::VectorFieldType VFT;
  typedef Albany::AbstractSTKFieldContainer::TensorFieldType TFT;

  // Utility constants
  const int numLayers  = params->get("NumLayers", 10);
  const int numNodes2d = nodes2d.size();
  const int numCells2d = cells2d.size();
  const bool ColOrdering  = params->get("Columnwise Ordering", false);

  stk::mesh::BulkData& bulkData2d = *basalMeshStruct->bulkData;
  stk::mesh::MetaData& metaData2d = *basalMeshStruct->metaData;

  int numNodeFields = node_fields_names.size();
  int numCellFields = cell_fields_names.size();
  int numScalars;
  double *values2d, *values3d;

  // Extrude node fields
  for (int ifield=0; ifield<numNodeFields; ++ifield)
  {
    switch (node_fields_ranks[ifield])
    {
      case 1:
      {
        *out << "Extruding Scalar Node field " << node_fields_names[ifield] << " from basal mesh\n";

        SFT* field2d = metaData2d.get_field<SFT>(stk::topology::NODE_RANK, node_fields_names[ifield]);
        SFT* field3d = metaData->get_field<SFT> (stk::topology::NODE_RANK, node_fields_names[ifield]);

        for (int inode=0; inode<numNodes2d; ++inode)
        {
          const stk::mesh::Entity& node2d = nodes2d[inode];
          stk::mesh::EntityId node2dId = bulkData2d.identifier(node2d) - 1;
          values2d = stk::mesh::field_data(*field2d,node2d);
          for (int il=0; il<=numLayers; ++il)
          {
            // Retrieve 3D node
            int node3dId = ColOrdering ? node2dId*(numLayers+1) + il + 1 : il*maxGlobalNodes2dId + node2dId + 1;
            stk::mesh::Entity node3d = bulkData->get_entity(stk::topology::NODE_RANK, node3dId);

            values3d = stk::mesh::field_data(*field3d,node3d);
            values3d[0] = values2d[0];
          }
        }
        break;
      }
      case 2:
      {
        *out << "Extruding Vector Node field " << node_fields_names[ifield] << " from basal mesh\n";

        VFT* field2d = metaData2d.get_field<VFT>(stk::topology::NODE_RANK, node_fields_names[ifield]);
        VFT* field3d = metaData->get_field<VFT> (stk::topology::NODE_RANK, node_fields_names[ifield]);
        for (int inode=0; inode<numNodes2d; ++inode)
        {
          const stk::mesh::Entity& node2d = nodes2d[inode];
          stk::mesh::EntityId node2dId = bulkData2d.identifier(node2d) - 1;
          values2d = stk::mesh::field_data(*field2d,node2d);
          numScalars = stk::mesh::field_scalars_per_entity(*field2d,node2d);
          for (int il=0; il<=numLayers; ++il)
          {
            // Retrieve 3D node
            int node3dId = ColOrdering ? node2dId*(numLayers+1)+il+1 : il*maxGlobalNodes2dId+node2dId + 1;
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
        *out << "Extruding Tensor Node field " << node_fields_names[ifield] << " from basal mesh\n";

        TFT* field2d = metaData2d.get_field<TFT>(stk::topology::NODE_RANK, node_fields_names[ifield]);
        TFT* field3d = metaData->get_field<TFT> (stk::topology::NODE_RANK, node_fields_names[ifield]);
        for (int inode=0; inode<numNodes2d; ++inode)
        {
          const stk::mesh::Entity& node2d = nodes2d[inode];
          stk::mesh::EntityId node2dId = bulkData2d.identifier(node2d) - 1;
          values2d = stk::mesh::field_data(*field2d,node2d);
          numScalars = stk::mesh::field_scalars_per_entity(*field2d,node2d);
          for (int il=0; il<=numLayers; ++il)
          {
            // Retrieve 3D node
            int node3dId = ColOrdering ? node2dId*(numLayers+1)+il+1 : il*maxGlobalNodes2dId+node2dId + 1;
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
  }

  // Extrude cell fields
  for (int ifield=0; ifield<numCellFields; ++ifield)
  {
    switch (cell_fields_ranks[ifield])
    {
      case 1:
      {
        *out << "Extruding Scalar Elem field " << cell_fields_names[ifield] << " from basal mesh\n";

        SFT* field2d = metaData2d.get_field<SFT>(stk::topology::ELEMENT_RANK, cell_fields_names[ifield]);
        SFT* field3d = metaData->get_field<SFT> (stk::topology::ELEMENT_RANK, cell_fields_names[ifield]);
        for (int icell=0; icell<numCells2d; ++icell)
        {
          const stk::mesh::Entity& cell2d = cells2d[icell];
          stk::mesh::EntityId cell2dId = bulkData2d.identifier(cell2d) - 1;
          values2d = stk::mesh::field_data(*field2d,cell2d);
          for (int il=0; il<numLayers; ++il)
          {
            // Retrieving the id of the 3d cells
            stk::mesh::EntityId prismId = ColOrdering ? numLayers*cell2dId + il + 1 : maxGlobalCells2dId*il + cell2dId + 1;
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
            }
          }
        }
        break;
      }
      case 2:
      {
        *out << "Extruding Vector Elem field " << cell_fields_names[ifield] << " from basal mesh\n";

        VFT* field2d = metaData2d.get_field<VFT>(stk::topology::ELEMENT_RANK, cell_fields_names[ifield]);
        VFT* field3d = metaData->get_field<VFT> (stk::topology::ELEMENT_RANK, cell_fields_names[ifield]);
        for (int icell=0; icell<numCells2d; ++icell)
        {
          const stk::mesh::Entity& cell2d = cells2d[icell];
          stk::mesh::EntityId cell2dId = bulkData2d.identifier(cell2d);
          numScalars = stk::mesh::field_scalars_per_entity(*field2d,cell2d);
          values2d = stk::mesh::field_data(*field2d,cell2d);
          for (int il=0; il<numLayers; ++il)
          {
            // Retrieving the id of the 3d cells
            stk::mesh::EntityId prismId = ColOrdering ? numLayers*(cell2dId-1)+il+1 : maxGlobalCells2dId*il+cell2dId;
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
        *out << "Extruding Tensor Elem field " << cell_fields_names[ifield] << " from basal mesh\n";

        TFT* field2d = metaData2d.get_field<TFT>(stk::topology::ELEMENT_RANK, cell_fields_names[ifield]);
        TFT* field3d = metaData->get_field<TFT> (stk::topology::ELEMENT_RANK, cell_fields_names[ifield]);
        for (int icell=0; icell<numCells2d; ++icell)
        {
          const stk::mesh::Entity& cell2d = cells2d[icell];
          stk::mesh::EntityId cell2dId = bulkData2d.identifier(cell2d);
          numScalars = stk::mesh::field_scalars_per_entity(*field2d,cell2d);
          values2d = stk::mesh::field_data(*field2d,cell2d);
          for (int il=0; il<numLayers; ++il)
          {
            // Retrieving the id of the 3d cells
            stk::mesh::EntityId prismId = ColOrdering ? numLayers*(cell2dId-1)+il+1 : maxGlobalCells2dId*il+cell2dId;
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
  validPL->set<std::string>("Surface Height File Name", "surface_height.ascii", "Name of the file containing the surface height data");
  validPL->set<std::string>("Thickness File Name", "thickness.ascii", "Name of the file containing the thickness data");
  validPL->set<std::string>("BedTopography File Name", "bed_topography.ascii", "Name of the file containing the bed topography data");
  validPL->set<std::string>("Surface Velocity File Name", "surface_velocity.ascii", "Name of the file containing the surface velocity data");
  validPL->set<std::string>("Surface Velocity RMS File Name", "velocity_RMS.ascii", "Name of the file containing the surface velocity RMS data");
  validPL->set<std::string>("Basal Friction File Name", "basal_friction.ascii", "Name of the file containing the basal friction data");
  validPL->set<std::string>("Temperature File Name", "temperature.ascii", "Name of the file containing the temperature data");
  validPL->set<std::string>("Element Shape", "Hexahedron", "Shape of the Element: Tetrahedron, Wedge, Hexahedron");
  validPL->set<int>("NumLayers", 10, "Number of vertical Layers of the extruded mesh. In a vertical column, the mesh will have numLayers+1 nodes");
  validPL->set<bool>("Use Glimmer Spacing", false, "When true, the layer spacing is computed according to Glimmer formula (layers are denser close to the bedrock)");
  validPL->set<bool>("Columnwise Ordering", false, "True for Columnwise ordering, false for Layerwise ordering");

  validPL->set<double>("Constant Surface Height",1.0,"Uniform surface height");
  validPL->set<double>("Constant Thickness",1.0,"Uniform thickness");

#ifdef ALBANY_FELIX
  validPL->sublist("Side Set Discretizations", false, "A sublist containing info for storing side discretizations");
#endif

  return validPL;
}
