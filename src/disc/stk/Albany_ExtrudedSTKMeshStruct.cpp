//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>

#include "Albany_ExtrudedSTKMeshStruct.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_IossSTKMeshStruct.hpp"
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

Albany::ExtrudedSTKMeshStruct::ExtrudedSTKMeshStruct(const Teuchos::RCP<Teuchos::ParameterList>& params, const Teuchos::RCP<const Teuchos_Comm>& comm) :
    GenericSTKMeshStruct(params, Teuchos::null, 3), out(Teuchos::VerboseObjectBase::getDefaultOStream()), periodic(false) {
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

  Teuchos::RCP<Teuchos::ParameterList> params2D(new Teuchos::ParameterList());
  params2D->set("Use Serial Mesh", params->get("Use Serial Mesh", false));
#ifdef ALBANY_SEACAS
  params2D->set("Exodus Input File Name", params->get("Exodus Input File Name", "IceSheet.exo"));
  if (params->isSublist("Side Sets Output"))
  {
    params2D->set<std::string>("Exodus Output File Name",params->sublist("Side Sets Output").sublist("basalside").get<std::string>("Exodus Output File Name"));
  }

  sideSetMeshStructs["basalside"] = Teuchos::rcp(new Albany::IossSTKMeshStruct(params2D, adaptParams, comm));

  Teuchos::RCP<Albany::StateInfoStruct> sis = Teuchos::rcp(new Albany::StateInfoStruct);
  Albany::AbstractFieldContainer::FieldContainerRequirements req;
#else
    // Above block of code could allow for 2D mesh to come from other sources instead of Ioss
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
              std::endl << "Error in ExtrudedSTKMeshStruct: Currently Requires 2D mesh to come from exodus");
#endif

  int ws_size = sideSetMeshStructs["basalside"]->getMeshSpecs()[0]->worksetSize;
  sideSetMeshStructs["basalside"]->setFieldAndBulkData(comm, params, 1, req, sis, ws_size);

  stk::mesh::Selector select_owned_in_part = stk::mesh::Selector(sideSetMeshStructs["basalside"]->metaData->universal_part()) & stk::mesh::Selector(sideSetMeshStructs["basalside"]->metaData->locally_owned_part());
  int numCells = stk::mesh::count_selected_entities(select_owned_in_part, sideSetMeshStructs["basalside"]->bulkData->buckets(stk::topology::ELEMENT_RANK));

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

  std::string elem2d_name(sideSetMeshStructs["basalside"]->getMeshSpecs()[0]->ctd.base->name);
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
  int worksetSizeMax = params->get("Workset Size", 50);
  int worksetSize = this->computeWorksetSize(worksetSizeMax, numCells);

  const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();

  this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub, nsNames, ssNames, worksetSize, partVec[0]->name(), ebNameToIndex, this->interleavedOrdering));

  // Add side set mesh specs
  this->meshSpecs[0]->sideSetMeshSpecs["basalside"] = sideSetMeshStructs["basalside"]->getMeshSpecs();
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
    const unsigned int worksetSize)
{
  int numLayers = params->get("NumLayers", 10);
  bool useGlimmerSpacing = params->get("Use Glimmer Spacing", false);
  GO maxGlobalElements2D = 0;
  GO maxGlobalVertices2dId = 0;
  GO numGlobalVertices2D = 0;
  GO maxGlobalEdges2D = 0;
  bool Ordering = params->get("Columnwise Ordering", false);
  bool isTetra = true;

  stk::mesh::BulkData& bulkData2D = *sideSetMeshStructs["basalside"]->bulkData;
  stk::mesh::MetaData& metaData2D = *sideSetMeshStructs["basalside"]->metaData; //bulkData2D.mesh_meta_data();

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
  //Teuchos::RCP<const Tpetra_Map> serial_nodes_map = Tpetra::createUniformContigMap<LO, GO>(numMyElements, comm);
  Teuchos::RCP<const Tpetra_Map> serial_nodes_map = Teuchos::rcp(new const Tpetra_Map(INVALID, numMyElements, 0, comm));
  Teuchos::RCP<Tpetra_Import> importOperator = Teuchos::rcp(new Tpetra_Import(serial_nodes_map, nodes_map));

  Teuchos::RCP<Tpetra_Vector> temp = Teuchos::rcp(new Tpetra_Vector(serial_nodes_map));
  Teuchos::RCP<Tpetra_Vector> sHeightVec;
  Teuchos::RCP<Tpetra_Vector> thickVec;
  Teuchos::RCP<Tpetra_Vector> bFrictionVec;
  Teuchos::RCP<Tpetra_MultiVector> temperatureVecInterp;
  Teuchos::RCP<Tpetra_MultiVector> sVelocityVec;
  Teuchos::RCP<Tpetra_MultiVector> velocityRMSVec;



  bool hasSurface_height =  std::find(req.begin(), req.end(), "surface_height") != req.end();

  {
    sHeightVec = Teuchos::rcp(new Tpetra_Vector(nodes_map));
    std::string fname = params->get<std::string>("Surface Height File Name", "surface_height.ascii");
    read2DFileSerial(fname, temp, comm);
    sHeightVec->doImport(*temp, *importOperator, Tpetra::INSERT);
  }
  Teuchos::ArrayRCP<const ST> sHeightVec_constView = sHeightVec->get1dView();


  bool hasThickness =  std::find(req.begin(), req.end(), "thickness") != req.end();

  {
    std::string fname = params->get<std::string>("Thickness File Name", "thickness.ascii");
    read2DFileSerial(fname, temp, comm);
    thickVec = Teuchos::rcp(new Tpetra_Vector(nodes_map));
    thickVec->doImport(*temp, *importOperator, Tpetra::INSERT);
  }
  Teuchos::ArrayRCP<const ST> thickVec_constView = thickVec->get1dView();


  bool hasBasal_friction = std::find(req.begin(), req.end(), "basal_friction") != req.end();
  if(hasBasal_friction)
  {
    bFrictionVec = Teuchos::rcp(new Tpetra_Vector(nodes_map));

    if (params->isParameter("Basal Friction File Name"))
    {
      std::string fname = params->get<std::string>("Basal Friction File Name", "basal_friction.ascii");
      read2DFileSerial(fname, temp, comm);
      bFrictionVec->doImport(*temp, *importOperator, Tpetra::INSERT);
    }
    else
    {
      // Try to load it from the 2D mesh
      Albany::AbstractSTKFieldContainer::ScalarFieldType* field = 0;
      field = metaData2D.get_field<Albany::AbstractSTKFieldContainer::ScalarFieldType>(stk::topology::NODE_RANK, "basal_friction");
      if (field!=0)
      {
        Teuchos::ArrayRCP<ST> bFrictionVec_view = bFrictionVec->get1dViewNonConst();

        stk::mesh::Entity node;
        stk::mesh::EntityId nodeId;
        int lid;
        double* values;

        //Now we have to stuff the vector in the mesh data
        for (int i(0); i<nodes2D.size(); ++i)
        {
          nodeId = bulkData2D.identifier(nodes2D[i]) - 1;
          lid    = nodes_map->getLocalElement((GO)(nodeId));

          values = stk::mesh::field_data(*field, nodes2D[i]);
          bFrictionVec_view[lid] = values[0];
        }
      }
      else
      {
        // We use a zero vector, but we issue a warning. Just in case the user forgot to setup something
        std::cout << "No file name specified for 'basal_friction', and no field retrieved from the mesh. Using a zero vector.\n";
      }
    }
  }

  bool hasTemperature = std::find(req.begin(), req.end(), "temperature") != req.end();
  if(hasTemperature) {
    Teuchos::RCP<Tpetra_MultiVector> temperatureVec;
    temperatureVecInterp = Teuchos::rcp(new Tpetra_MultiVector(nodes_map, numLayers + 1));
    std::string fname = params->get<std::string>("Temperature File Name", "temperature.ascii");
    readFileSerial(fname, serial_nodes_map, nodes_map, importOperator, temperatureVec, temperatureNormalizedZ, comm);


   int il0, il1, verticalTSize(temperatureVec->getNumVectors());
    double h0(0.0);

    for (int il = 0; il < numLayers + 1; il++) {
      if (levelsNormalizedThickness[il] <= temperatureNormalizedZ[0]) {
        il0 = 0;
        il1 = 0;
        h0 = 1.0;
      }

      else if (levelsNormalizedThickness[il] >= temperatureNormalizedZ[verticalTSize - 1]) {
        il0 = verticalTSize - 1;
        il1 = verticalTSize - 1;
        h0 = 0.0;
      }

      else {
        int k = 0;
        while (levelsNormalizedThickness[il] > temperatureNormalizedZ[++k])
          ;
        il0 = k - 1;
        il1 = k;
        h0 = (temperatureNormalizedZ[il1] - levelsNormalizedThickness[il]) / (temperatureNormalizedZ[il1] - temperatureNormalizedZ[il0]);
      }
      Teuchos::ArrayRCP<ST> temperatureVecInterp_nonConstView = temperatureVecInterp->getVectorNonConst(il)->get1dViewNonConst();
      Teuchos::ArrayRCP<const ST> temperatureVec_constView_il0 = temperatureVec->getVectorNonConst(il0)->get1dView();
      Teuchos::ArrayRCP<const ST> temperatureVec_constView_il1 = temperatureVec->getVectorNonConst(il1)->get1dView();

      for (int i = 0; i < nodes_map->getNodeNumElements(); i++)
        temperatureVecInterp_nonConstView[i] = h0 * temperatureVec_constView_il0[i] + (1.0 - h0) * temperatureVec_constView_il1[i];
    }
  }

  Tpetra_MultiVector tempSV(serial_nodes_map,neq_);

  bool hasSurfaceVelocity = std::find(req.begin(), req.end(), "surface_velocity") != req.end();
  if(hasSurfaceVelocity) {
    std::string fname = params->get<std::string>("Surface Velocity File Name", "surface_velocity.ascii");
    readFileSerial(fname, tempSV, comm);
    sVelocityVec = Teuchos::rcp(new Tpetra_MultiVector (nodes_map, neq_));
    sVelocityVec->doImport(tempSV, *importOperator, Tpetra::INSERT);
  }

  bool hasSurfaceVelocityRMS = std::find(req.begin(), req.end(), "surface_velocity_rms") != req.end();
  if(hasSurfaceVelocityRMS) {
    std::string fname = params->get<std::string>("Surface Velocity RMS File Name", "velocity_RMS.ascii");
    readFileSerial(fname, tempSV, comm);
    velocityRMSVec = Teuchos::rcp(new Tpetra_MultiVector (nodes_map, neq_));
    velocityRMSVec->doImport(tempSV, *importOperator, Tpetra::INSERT);
  }

  if (comm->getRank() == 0) std::cout << " done." << std::endl;

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

  this->SetupFieldData(comm, neq_, req, sis, worksetSize);

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

  AbstractSTKFieldContainer::IntScalarFieldType* proc_rank_field = fieldContainer->getProcRankField();
  VectorFieldType* coordinates_field = fieldContainer->getCoordinatesField();
  stk::mesh::FieldBase const* coordinates_field2d = metaData2D.coordinate_field();
  VectorFieldType* surface_velocity_field = metaData->get_field<VectorFieldType>(stk::topology::NODE_RANK, "surface_velocity");
  VectorFieldType* surface_velocity_RMS_field = metaData->get_field<VectorFieldType>(stk::topology::NODE_RANK, "surface_velocity_rms");
  ScalarFieldType* surface_height_field = metaData->get_field<ScalarFieldType>(stk::topology::NODE_RANK, "surface_height");
  ScalarFieldType* thickness_field = metaData->get_field<ScalarFieldType>(stk::topology::NODE_RANK, "thickness");
  ScalarFieldType* basal_friction_field = metaData->get_field<ScalarFieldType>(stk::topology::NODE_RANK, "basal_friction");
  ElemScalarFieldType* temperature_field = metaData->get_field<ElemScalarFieldType>(stk::topology::ELEMENT_RANK, "temperature");

  std::vector<GO> prismMpasIds(NumBaseElemeNodes), prismGlobalIds(2 * NumBaseElemeNodes);

  for (int i = 0; i < (numLayers + 1) * nodes2D.size(); i++) {
    int ib = (Ordering == 0) * (i % lVertexColumnShift) + (Ordering == 1) * (i / vertexLayerShift);
    int il = (Ordering == 0) * (i / lVertexColumnShift) + (Ordering == 1) * (i % vertexLayerShift);
    stk::mesh::Entity node;
    stk::mesh::Entity node2d = nodes2D[ib];
    stk::mesh::EntityId node2dId = bulkData2D.identifier(node2d) - 1;
    if (il == 0)
      node = bulkData->declare_entity(stk::topology::NODE_RANK, il * vertexColumnShift + vertexLayerShift * node2dId + 1, singlePartVec);
    else
      node = bulkData->declare_entity(stk::topology::NODE_RANK, il * vertexColumnShift + vertexLayerShift * node2dId + 1, nodePartVec);

    std::vector<int> sharing_procs;
    bulkData2D.comm_shared_procs( bulkData2D.entity_key(node2d), sharing_procs );
    for(int iproc=0; iproc<sharing_procs.size(); ++iproc)
      bulkData->add_node_sharing(node, sharing_procs[iproc]);


    double* coord = stk::mesh::field_data(*coordinates_field, node);
    double const* coord2d = (double const*) stk::mesh::field_data(*coordinates_field2d, node2d);
    coord[0] = coord2d[0];
    coord[1] = coord2d[1];

    int lid = nodes_map->getLocalElement((GO)(node2dId));
    coord[2] = sHeightVec_constView[lid] - thickVec_constView[lid] * (1. - levelsNormalizedThickness[il]);

    if(hasSurface_height && surface_height_field) {
      double* sHeight = stk::mesh::field_data(*surface_height_field, node);
      sHeight[0] = sHeightVec_constView[lid];
    }

    if(hasThickness && thickness_field) {
      double* thick = stk::mesh::field_data(*thickness_field, node);
      thick[0] = thickVec_constView[lid];
    }

    if(surface_velocity_field) {
      double* sVelocity = stk::mesh::field_data(*surface_velocity_field, node);
      Teuchos::ArrayRCP<const ST> sVelocityVec_constView_0 = sVelocityVec->getVectorNonConst(0)->get1dView();
      Teuchos::ArrayRCP<const ST> sVelocityVec_constView_1 = sVelocityVec->getVectorNonConst(1)->get1dView();
      sVelocity[0] = sVelocityVec_constView_0[lid];
      sVelocity[1] = sVelocityVec_constView_1[lid];
    }

    if(surface_velocity_RMS_field) {
      double* velocityRMS = stk::mesh::field_data(*surface_velocity_RMS_field, node);
      Teuchos::ArrayRCP<const ST> velocityRMSVec_constView_0 = velocityRMSVec->getVectorNonConst(0)->get1dView();
      Teuchos::ArrayRCP<const ST> velocityRMSVec_constView_1 = velocityRMSVec->getVectorNonConst(1)->get1dView();
      velocityRMS[0] = velocityRMSVec_constView_0[lid];
      velocityRMS[1] = velocityRMSVec_constView_1[lid];

    }

    if(hasBasal_friction && basal_friction_field) {
      double* bFriction = stk::mesh::field_data(*basal_friction_field, node);
      bFriction[0] = bFrictionVec->get1dView()[lid];
    }
  }

  GO tetrasLocalIdsOnPrism[3][4];

  for (int i = 0; i < cells2D.size() * numLayers; i++) {

    int ib = (Ordering == 0) * (i % lElemColumnShift) + (Ordering == 1) * (i / elemLayerShift);
    int il = (Ordering == 0) * (i / lElemColumnShift) + (Ordering == 1) * (i % elemLayerShift);

    int shift = il * vertexColumnShift;

    singlePartVec[0] = partVec[ebNo];

    //TODO: this could be done only in the first layer and then copied into the other layers

    stk::mesh::Entity const* rel = bulkData2D.begin_nodes(cells2D[ib]);
    double tempOnPrism = 0; //Set temperature constant on each prism/Hexa
    Teuchos::ArrayRCP<const ST> temperatureVecInterp_constView_il = temperatureVecInterp->getVectorNonConst(il)->get1dView();
    Teuchos::ArrayRCP<const ST> temperatureVecInterp_constView_ilplus1 = temperatureVecInterp->getVectorNonConst(il + 1)->get1dView();
    for (int j = 0; j < NumBaseElemeNodes; j++) {
      stk::mesh::EntityId node2dId = bulkData2D.identifier(rel[j]) - 1;
      int node2dLId = nodes_map->getLocalElement((GO)node2dId);
      stk::mesh::EntityId mpasLowerId = vertexLayerShift * node2dId;
      stk::mesh::EntityId lowerId = shift + vertexLayerShift * node2dId;
      prismMpasIds[j] = mpasLowerId;
      prismGlobalIds[j] = lowerId;
      prismGlobalIds[j + NumBaseElemeNodes] = lowerId + vertexColumnShift;
      if(hasTemperature)
        tempOnPrism += 1. / NumBaseElemeNodes / 2. * (temperatureVecInterp_constView_il[node2dLId] + temperatureVecInterp_constView_ilplus1[node2dLId]);
    }

    switch (ElemShape) {
    case Tetrahedron: {
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
        if(hasTemperature && temperature_field) {
          double* temperature = stk::mesh::field_data(*temperature_field, elem);
          temperature[0] = tempOnPrism;
        }
      }
    }
      break;
    case Wedge:
    case Hexahedron: {
      stk::mesh::EntityId prismId = il * elemColumnShift + elemLayerShift * (bulkData2D.identifier(cells2D[ib]) - 1);
      stk::mesh::Entity elem = bulkData->declare_entity(stk::topology::ELEMENT_RANK, prismId + 1, singlePartVec);
      for (int j = 0; j < 2 * NumBaseElemeNodes; j++) {
        stk::mesh::Entity node = bulkData->get_entity(stk::topology::NODE_RANK, prismGlobalIds[j] + 1);
        bulkData->declare_relation(elem, node, j);
      }
      int* p_rank = (int*) stk::mesh::field_data(*proc_rank_field, elem);
      p_rank[0] = comm->getRank();
      if(hasTemperature && temperature_field) {
        double* temperature = stk::mesh::field_data(*temperature_field, elem);
        temperature[0] = tempOnPrism;
      }
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

      stk::mesh::Entity side0 = bulkData->declare_entity(metaData->side_rank(), 2 * edgeColumnShift * il +  2 * Edge2dId * edgeLayerShift + 1, singlePartVec);
      stk::mesh::Entity side1 = bulkData->declare_entity(metaData->side_rank(), 2 * edgeColumnShift * il +  2 * Edge2dId * edgeLayerShift + 1 + 1, singlePartVec);

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
      stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), edgeColumnShift * il +Edge2dId * edgeLayerShift + 1, singlePartVec);
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

  //then we store the lower and upper faces of prisms, which corresponds to triangles of the basal mesh
  edgeLayerShift = (Ordering == 0) ? 1 : numLayers + 1;
  edgeColumnShift = elemColumnShift;

  singlePartVec[0] = ssPartVec["basalside"];


  GO edgeOffset = maxGlobalEdges2D * numLayers;
  if(ElemShape == Tetrahedron) edgeOffset *= 2;

  for (int i = 0; i < cells2D.size(); i++) {
    stk::mesh::Entity elem2d = cells2D[i];
    stk::mesh::EntityId elem2d_id = bulkData2D.identifier(elem2d) - 1;
    stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), elem2d_id + edgeOffset + 1, singlePartVec);
    stk::mesh::Entity elem = bulkData->get_entity(stk::topology::ELEMENT_RANK, elem2d_id * numSubelemOnPrism * elemLayerShift + 1);
    bulkData->declare_relation(elem, side, basalSideLID);

    stk::mesh::Entity const* rel_elemNodes = bulkData->begin_nodes(elem);
    for (int j = 0; j < numBasalSidePoints; j++) {
      stk::mesh::Entity node = rel_elemNodes[this->meshSpecs[0]->ctd.side[basalSideLID].node[j]];
      bulkData->declare_relation(side, node, j);
    }
  }

  singlePartVec[0] = ssPartVec["upperside"];

  edgeOffset += maxGlobalElements2D;

  for (int i = 0; i < cells2D.size(); i++) {
    stk::mesh::Entity elem2d = cells2D[i];
    stk::mesh::EntityId elem2d_id = bulkData2D.identifier(elem2d) - 1;
    stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), elem2d_id  + edgeOffset + 1, singlePartVec);
    stk::mesh::Entity elem = bulkData->get_entity(stk::topology::ELEMENT_RANK, elem2d_id * numSubelemOnPrism * elemLayerShift + (numLayers - 1) * numSubelemOnPrism * elemColumnShift + 1 + (numSubelemOnPrism - 1));
    bulkData->declare_relation(elem, side, upperSideLID);

    stk::mesh::Entity const* rel_elemNodes = bulkData->begin_nodes(elem);
    for (int j = 0; j < numBasalSidePoints; j++) {
      stk::mesh::Entity node = rel_elemNodes[this->meshSpecs[0]->ctd.side[upperSideLID].node[j]];
      bulkData->declare_relation(side, node, j);
    }
  }
  //Albany::fix_node_sharing(*bulkData);
  bulkData->modification_end();

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

Teuchos::RCP<const Teuchos::ParameterList> Albany::ExtrudedSTKMeshStruct::getValidDiscretizationParameters() const {
  Teuchos::RCP<Teuchos::ParameterList> validPL = this->getValidGenericSTKParameters("Valid Extruded_DiscParams");
  validPL->set<bool>("Export 2D Data", "", "If true, exports the 2D mesh in GMSH format");
  validPL->set<std::string>("GMSH 2D Output File Name", "", "File Name for GMSH 2D Basal Mesh Export");
  validPL->set<std::string>("Exodus Input File Name", "", "File Name For Exodus Mesh Input");
  validPL->set<std::string>("Surface Height File Name", "surface_height.ascii", "Name of the file containing the surface height data");
  validPL->set<std::string>("Thickness File Name", "thickness.ascii", "Name of the file containing the thickness data");
  validPL->set<std::string>("Surface Velocity File Name", "surface_velocity.ascii", "Name of the file containing the surface velocity data");
  validPL->set<std::string>("Surface Velocity RMS File Name", "velocity_RMS.ascii", "Name of the file containing the surface velocity RMS data");
  validPL->set<std::string>("Basal Friction File Name", "basal_friction.ascii", "Name of the file containing the basal friction data");
  validPL->set<std::string>("Temperature File Name", "temperature.ascii", "Name of the file containing the temperature data");
  validPL->set<std::string>("Element Shape", "Hexahedron", "Shape of the Element: Tetrahedron, Wedge, Hexahedron");
  validPL->set<int>("NumLayers", 10, "Number of vertical Layers of the extruded mesh. In a vertical column, the mesh will have numLayers+1 nodes");
  validPL->set<bool>("Use Glimmer Spacing", false, "When true, the layer spacing is computed according to Glimmer formula (layers are denser close to the bedrock)");
  validPL->set<bool>("Columnwise Ordering", false, "True for Columnwise ordering, false for Layerwise ordering");

#ifdef ALBANY_FELIX
  validPL->sublist("Side Sets Output", false, "A sublist containing info for storing side set meshes");
#endif

  return validPL;
}

void Albany::ExtrudedSTKMeshStruct::read2DFileSerial(std::string &fname, Teuchos::RCP<Tpetra_Vector> content, const Teuchos::RCP<const Teuchos_Comm>& comm) {
  GO numNodes;
  Teuchos::ArrayRCP<ST> content_nonConstView = content->get1dViewNonConst();
  if (comm->getRank() == 0) {
    std::ifstream ifile;
    ifile.open(fname.c_str());
    if (ifile.is_open()) {
      ifile >> numNodes;
      TEUCHOS_TEST_FOR_EXCEPTION(numNodes != content->getLocalLength(), Teuchos::Exceptions::InvalidParameterValue, std::endl << "Error in ExtrudedSTKMeshStruct: Number of nodes in file " << fname << " (" << numNodes << ") is different from the number expected (" << content->getLocalLength() << ")" << std::endl);

      for (GO i = 0; i < numNodes; i++)
        ifile >> content_nonConstView[i];
      ifile.close();
    } else {
      std::cout << "Warning in ExtrudedSTKMeshStruct: Unable to open the file " << fname << std::endl;
    }
  }
}

void Albany::ExtrudedSTKMeshStruct::readFileSerial(std::string &fname, Tpetra_MultiVector& contentVec, const Teuchos::RCP<const Teuchos_Comm>& comm) {
  GO numNodes, numComponents;
  if (comm->getRank() == 0) {
    std::ifstream ifile;
    ifile.open(fname.c_str());
    if (ifile.is_open()) {
      ifile >> numNodes >> numComponents;
      TEUCHOS_TEST_FOR_EXCEPTION(numNodes != contentVec.getLocalLength(), Teuchos::Exceptions::InvalidParameterValue,
          std::endl << "Error in ExtrudedSTKMeshStruct: Number of nodes in file " << fname << " (" << numNodes << ") is different from the number expected (" << contentVec.getLocalLength() << ")" << std::endl);
      TEUCHOS_TEST_FOR_EXCEPTION(numComponents > contentVec.getNumVectors(), Teuchos::Exceptions::InvalidParameterValue,
          std::endl << "Error in ExtrudedSTKMeshStruct: Number of components in file " << fname << " (" << numComponents << ") is different from the number expected (" << contentVec.getNumVectors() << ")" << std::endl);
      for (int il = 0; il < numComponents; ++il) {
        Teuchos::ArrayRCP<ST> contentVec_nonConstView = contentVec.getVectorNonConst(il)->get1dViewNonConst();
        for (GO i = 0; i < numNodes; i++)
          ifile >> contentVec_nonConstView[i];
      }
      ifile.close();
    } else {
      std::cout << "Warning in ExtrudedSTKMeshStruct: Unable to open input file " << fname << std::endl;
      //  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
      //      std::endl << "Error in ExtrudedSTKMeshStruct: Unable to open input file " << fname << std::endl);
    }
  }
}

void Albany::ExtrudedSTKMeshStruct::readFileSerial(std::string &fname, Teuchos::RCP<const Tpetra_Map> map_serial, Teuchos::RCP<const Tpetra_Map> map, Teuchos::RCP<Tpetra_Import> importOperator, Teuchos::RCP<Tpetra_MultiVector>& temperatureVec, std::vector<double>& zCoords, const Teuchos::RCP<const Teuchos_Comm>& comm) {
  GO numNodes;
  int numComponents;
  std::ifstream ifile;
  if (comm->getRank() == 0) {

    ifile.open(fname.c_str());
    if (ifile.is_open()) {
      ifile >> numNodes >> numComponents;

    //  std::cout << "numNodes >> numComponents: " << numNodes << " " << numComponents << std::endl;

      TEUCHOS_TEST_FOR_EXCEPTION(numNodes != map_serial->getNodeNumElements(), Teuchos::Exceptions::InvalidParameterValue,
          std::endl << "Error in ExtrudedSTKMeshStruct: Number of nodes in file " << fname << " (" << numNodes << ") is different from the number expected (" << map_serial->getNodeNumElements() << ")" << std::endl);
    } else {
      std::cout << "Warning in ExtrudedSTKMeshStruct: Unable to open input file " << fname << std::endl;
      //  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
      //      std::endl << "Error in ExtrudedSTKMeshStruct: Unable to open input file " << fname << std::endl);
    }
  }
  // The first int is for Comm<int>; the second is the type of numComponents.
  Teuchos::broadcast<int, int>(*comm, 0, 1, &numComponents);
  zCoords.resize(numComponents);
  Tpetra_Vector tempT(map_serial);

  if (comm->getRank() == 0) {
    for (int i = 0; i < numComponents; ++i)
      ifile >> zCoords[i];
  }
  //comm->Broadcast(&zCoords[0], numComponents, 0);
  //IK, 10/1/14: double should be ST?
  Teuchos::broadcast<int, double>(*comm, 0, numComponents, &zCoords[0]);

  temperatureVec = Teuchos::rcp(new Tpetra_MultiVector(map, numComponents));

  Teuchos::ArrayRCP<ST> tempT_nonConstView = tempT.get1dViewNonConst();
  for (int il = 0; il < numComponents; ++il) {
    if (comm->getRank() == 0)
      for (GO i = 0; i < numNodes; i++)
        ifile >> tempT_nonConstView[i];
    temperatureVec->getVectorNonConst(il)->doImport(tempT, *importOperator, Tpetra::INSERT);
  }

  if (comm->getRank() == 0)
    ifile.close();


}
