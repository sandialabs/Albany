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
#include <stk_mesh/base/FieldData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <Epetra_Import.h>

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

//#include <stk_mesh/fem/FEMHelpers.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "Albany_Utils.hpp"

//TODO: Generalize the importer so that it can extrude quad meshes

Albany::ExtrudedSTKMeshStruct::ExtrudedSTKMeshStruct(const Teuchos::RCP<Teuchos::ParameterList>& params, const Teuchos::RCP<const Epetra_Comm>& comm) :
    GenericSTKMeshStruct(params, Teuchos::null, 3), out(Teuchos::VerboseObjectBase::getDefaultOStream()), periodic(false) {
  params->validateParameters(*getValidDiscretizationParameters(), 0);

  std::string ebn = "Element Block 0";
  partVec[0] = &metaData->declare_part(ebn, metaData->element_rank());
  ebNameToIndex[ebn] = 0;

#ifdef ALBANY_SEACAS
  stk_classic::io::put_io_part_attribute(*partVec[0]);
#endif

  std::vector<std::string> nsNames;
  std::string nsn = "Lateral";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = &metaData->declare_part(nsn, metaData->node_rank());
#ifdef ALBANY_SEACAS
  stk_classic::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn = "Internal";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = &metaData->declare_part(nsn, metaData->node_rank());
#ifdef ALBANY_SEACAS
  stk_classic::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn = "Bottom";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = &metaData->declare_part(nsn, metaData->node_rank());
#ifdef ALBANY_SEACAS
  stk_classic::io::put_io_part_attribute(*nsPartVec[nsn]);
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
  stk_classic::io::put_io_part_attribute(*ssPartVec[ssnLat]);
  stk_classic::io::put_io_part_attribute(*ssPartVec[ssnBottom]);
  stk_classic::io::put_io_part_attribute(*ssPartVec[ssnTop]);
#endif

  Teuchos::RCP<Teuchos::ParameterList> params2D(new Teuchos::ParameterList());
  params2D->set("Use Serial Mesh", params->get("Use Serial Mesh", false));
  params2D->set("Exodus Input File Name", params->get("Exodus Input File Name", "IceSheet.exo"));
  meshStruct2D = Teuchos::rcp(new Albany::IossSTKMeshStruct(params2D, adaptParams, comm));
  Teuchos::RCP<Albany::StateInfoStruct> sis = Teuchos::rcp(new Albany::StateInfoStruct);
  Albany::AbstractFieldContainer::FieldContainerRequirements req;
  meshStruct2D->setFieldAndBulkData(comm, params, 1, req, sis, meshStruct2D->getMeshSpecs()[0]->worksetSize);

  std::vector<stk_classic::mesh::Entity *> cells;
  stk_classic::mesh::Selector select_owned_in_part = stk_classic::mesh::Selector(meshStruct2D->metaData->universal_part()) & stk_classic::mesh::Selector(meshStruct2D->metaData->locally_owned_part());
  int numCells = stk_classic::mesh::count_selected_entities(select_owned_in_part, meshStruct2D->bulkData->buckets(meshStruct2D->metaData->element_rank()));

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

  std::string elem2d_name(meshStruct2D->getMeshSpecs()[0]->ctd.base->name);
  TEUCHOS_TEST_FOR_EXCEPTION(basalside_name != elem2d_name, Teuchos::Exceptions::InvalidParameterValue,
                std::endl << "Error in ExtrudedSTKMeshStruct: Expecting topology name of elements of 2d mesh to be " <<  basalside_name << " but it is " << elem2d_name);


  switch (ElemShape) {
  case Tetrahedron:
    stk_classic::mesh::fem::set_cell_topology<shards::Tetrahedron<4> >(*partVec[0]);
    stk_classic::mesh::fem::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnBottom]);
    stk_classic::mesh::fem::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnTop]);
    stk_classic::mesh::fem::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnLat]);
    NumBaseElemeNodes = 3;
    break;
  case Wedge:
    stk_classic::mesh::fem::set_cell_topology<shards::Wedge<6> >(*partVec[0]);
    stk_classic::mesh::fem::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnBottom]);
    stk_classic::mesh::fem::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnTop]);
    stk_classic::mesh::fem::set_cell_topology<shards::Quadrilateral<4> >(*ssPartVec[ssnLat]);
    NumBaseElemeNodes = 3;
    break;
  case Hexahedron:
    stk_classic::mesh::fem::set_cell_topology<shards::Hexahedron<8> >(*partVec[0]);
    stk_classic::mesh::fem::set_cell_topology<shards::Quadrilateral<4> >(*ssPartVec[ssnBottom]);
    stk_classic::mesh::fem::set_cell_topology<shards::Quadrilateral<4> >(*ssPartVec[ssnTop]);
    stk_classic::mesh::fem::set_cell_topology<shards::Quadrilateral<4> >(*ssPartVec[ssnLat]);
    NumBaseElemeNodes = 4;
    break;
  }



  numDim = 3;
  int cub = params->get("Cubature Degree", 3);
  int worksetSizeMax = params->get("Workset Size", 50);
  int worksetSize = this->computeWorksetSize(worksetSizeMax, numCells);

  const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();

  this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub, nsNames, ssNames, worksetSize, partVec[0]->name(), ebNameToIndex, this->interleavedOrdering));

}

Albany::ExtrudedSTKMeshStruct::~ExtrudedSTKMeshStruct() {
}

void Albany::ExtrudedSTKMeshStruct::setFieldAndBulkData(const Teuchos::RCP<const Epetra_Comm>& comm, const Teuchos::RCP<Teuchos::ParameterList>& params, const unsigned int neq_, const AbstractFieldContainer::FieldContainerRequirements& req, const Teuchos::RCP<Albany::StateInfoStruct>& sis,
    const unsigned int worksetSize) {

  int numLayers = params->get("NumLayers", 10);
  bool useGlimmerSpacing = params->get("Use Glimmer Spacing", false);
  long long int maxGlobalElements2D = 0;
  long long int maxGlobalVertices2dId = 0;
  long long int numGlobalVertices2D = 0;
  long long int maxGlobalEdges2D = 0;
  bool Ordering = params->get("Columnwise Ordering", false);
  bool isTetra = true;

  std::vector<double> levelsNormalizedThickness(numLayers + 1), temperatureNormalizedZ;

  if(useGlimmerSpacing)
    for (int i = 0; i < numLayers+1; i++)
      levelsNormalizedThickness[numLayers-i] = 1.0- (1.0 - std::pow(double(i) / numLayers + 1.0, -2))/(1.0 - std::pow(2.0, -2));
  else  //uniform layers
    for (int i = 0; i < numLayers+1; i++)
      levelsNormalizedThickness[i] = double(i) / numLayers;

  /*std::cout<< "Levels: ";
  for (int i = 0; i < numLayers+1; i++)
    std::cout<< levelsNormalizedThickness[i] << " ";
  std::cout<< "\n";*/

  stk_classic::mesh::Selector select_owned_in_part = stk_classic::mesh::Selector(meshStruct2D->metaData->universal_part()) & stk_classic::mesh::Selector(meshStruct2D->metaData->locally_owned_part());

  stk_classic::mesh::Selector select_overlap_in_part = stk_classic::mesh::Selector(meshStruct2D->metaData->universal_part()) & (stk_classic::mesh::Selector(meshStruct2D->metaData->locally_owned_part()) | stk_classic::mesh::Selector(meshStruct2D->metaData->globally_shared_part()));

  stk_classic::mesh::Selector select_edges = stk_classic::mesh::Selector(*meshStruct2D->metaData->get_part("LateralSide")) & (stk_classic::mesh::Selector(meshStruct2D->metaData->locally_owned_part()) | stk_classic::mesh::Selector(meshStruct2D->metaData->globally_shared_part()));

  std::vector<stk_classic::mesh::Entity *> cells;
  stk_classic::mesh::get_selected_entities(select_overlap_in_part, meshStruct2D->bulkData->buckets(meshStruct2D->metaData->element_rank()), cells);

  std::vector<stk_classic::mesh::Entity *> nodes;
  stk_classic::mesh::get_selected_entities(select_overlap_in_part, meshStruct2D->bulkData->buckets(meshStruct2D->metaData->node_rank()), nodes);

  std::vector<stk_classic::mesh::Entity *> edges;
  stk_classic::mesh::get_selected_entities(select_edges, meshStruct2D->bulkData->buckets(meshStruct2D->metaData->side_rank()), edges);

  long long int maxOwnedElements2D(0), maxOwnedNodes2D(0), maxOwnedSides2D(0), numOwnedNodes2D(0);
  for (int i = 0; i < cells.size(); i++)
    maxOwnedElements2D = std::max(maxOwnedElements2D, (long long int) cells[i]->identifier());
  for (int i = 0; i < nodes.size(); i++)
    maxOwnedNodes2D = std::max(maxOwnedNodes2D, (long long int) nodes[i]->identifier());
  for (int i = 0; i < edges.size(); i++)
    maxOwnedSides2D = std::max(maxOwnedSides2D, (long long int) edges[i]->identifier());
  numOwnedNodes2D = stk_classic::mesh::count_selected_entities(select_owned_in_part, meshStruct2D->bulkData->buckets(meshStruct2D->metaData->node_rank()));

  comm->MaxAll(&maxOwnedElements2D, &maxGlobalElements2D, 1);
  comm->MaxAll(&maxOwnedNodes2D, &maxGlobalVertices2dId, 1);
  comm->MaxAll(&maxOwnedSides2D, &maxGlobalEdges2D, 1);
  comm->SumAll(&numOwnedNodes2D, &numGlobalVertices2D, 1);


  if (comm->MyPID() == 0) std::cout << "Importing ascii files ...";

  //std::cout << "Num Global Elements: " << maxGlobalElements2D<< " " << maxGlobalVertices2dId<< " " << maxGlobalEdges2D << std::endl;

  std::vector<int> indices(nodes.size()), serialIndices;
  for (int i = 0; i < nodes.size(); ++i)
    indices[i] = nodes[i]->identifier() - 1;

  const Epetra_Map nodes_map(-1, indices.size(), &indices[0], 0, *comm);
  int numMyElements = (comm->MyPID() == 0) ? numGlobalVertices2D : 0;
  const Epetra_Map serial_nodes_map(-1, numMyElements, 0, *comm);
  Epetra_Import importOperator(nodes_map, serial_nodes_map);

  Epetra_Vector temp(serial_nodes_map);
  Teuchos::RCP<Epetra_Vector> sHeightVec;
  Teuchos::RCP<Epetra_Vector> thickVec;
  Teuchos::RCP<Epetra_Vector> bFrictionVec;
  Teuchos::RCP<std::vector<Epetra_Vector> > temperatureVecInterp;
  Teuchos::RCP<std::vector<Epetra_Vector> > sVelocityVec;
  Teuchos::RCP<std::vector<Epetra_Vector> > velocityRMSVec;


  bool hasSurfaceHeight =  std::find(req.begin(), req.end(), "Surface Height") != req.end();

  {
    sHeightVec = Teuchos::rcp(new Epetra_Vector(nodes_map));
    std::string fname = params->get<std::string>("Surface Height File Name", "surface_height.ascii");
    read2DFileSerial(fname, temp, comm);
    sHeightVec->Import(temp, importOperator, Insert);
  }


  bool hasThickness =  std::find(req.begin(), req.end(), "Thickness") != req.end();

  {
    std::string fname = params->get<std::string>("Thickness File Name", "thickness.ascii");
    read2DFileSerial(fname, temp, comm);
    thickVec = Teuchos::rcp(new Epetra_Vector(nodes_map));
    thickVec->Import(temp, importOperator, Insert);
  }


  bool hasBasalFriction = std::find(req.begin(), req.end(), "Basal Friction") != req.end();
  if(hasBasalFriction) {
    std::string fname = params->get<std::string>("Basal Friction File Name", "basal_friction.ascii");
    read2DFileSerial(fname, temp, comm);
    bFrictionVec = Teuchos::rcp(new Epetra_Vector(nodes_map));
    bFrictionVec->Import(temp, importOperator, Insert);
  }

  bool hasTemperature = std::find(req.begin(), req.end(), "Temperature") != req.end();
  if(hasTemperature) {
    std::vector<Epetra_Vector> temperatureVec;
    temperatureVecInterp = Teuchos::rcp(new std::vector<Epetra_Vector>(numLayers + 1, Epetra_Vector(nodes_map)));
    std::string fname = params->get<std::string>("Temperature File Name", "temperature.ascii");
    readFileSerial(fname, serial_nodes_map, nodes_map, importOperator, temperatureVec, temperatureNormalizedZ, comm);


    int il0, il1, verticalTSize(temperatureVec.size());
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

      for (int i = 0; i < nodes_map.NumMyElements(); i++)
        (*temperatureVecInterp)[il][i] = h0 * temperatureVec[il0][i] + (1.0 - h0) * temperatureVec[il1][i];
    }
  }

  std::vector<Epetra_Vector> tempSV(neq_, Epetra_Vector(serial_nodes_map));

  bool hasSurfaceVelocity = std::find(req.begin(), req.end(), "Surface Velocity") != req.end();
  if(hasSurfaceVelocity) {
    std::string fname = params->get<std::string>("Surface Velocity File Name", "surface_velocity.ascii");
    readFileSerial(fname, tempSV, comm);
    sVelocityVec = Teuchos::rcp(new std::vector<Epetra_Vector> (neq_, Epetra_Vector(nodes_map)));
    for (int i = 0; i < tempSV.size(); i++)
      (*sVelocityVec)[i].Import(tempSV[i], importOperator, Insert);
  }

  bool hasVelocityRMS = std::find(req.begin(), req.end(), "Velocity RMS") != req.end();
  if(hasVelocityRMS) {
    std::string fname = params->get<std::string>("Velocity RMS File Name", "velocity_RMS.ascii");
    readFileSerial(fname, tempSV, comm);
    velocityRMSVec = Teuchos::rcp(new std::vector<Epetra_Vector> (neq_, Epetra_Vector(nodes_map)));
    for (int i = 0; i < tempSV.size(); i++)
      (*velocityRMSVec)[i].Import(tempSV[i], importOperator, Insert);
  }

  if (comm->MyPID() == 0) std::cout << " done." << std::endl;

  long long int elemColumnShift = (Ordering == 1) ? 1 : maxGlobalElements2D;
  int lElemColumnShift = (Ordering == 1) ? 1 : cells.size();
  int elemLayerShift = (Ordering == 0) ? 1 : numLayers;

  long long int vertexColumnShift = (Ordering == 1) ? 1 : maxGlobalVertices2dId;
  int lVertexColumnShift = (Ordering == 1) ? 1 : nodes.size();
  int vertexLayerShift = (Ordering == 0) ? 1 : numLayers + 1;

  long long int edgeColumnShift = (Ordering == 1) ? 1 : maxGlobalEdges2D;
  int lEdgeColumnShift = (Ordering == 1) ? 1 : edges.size();
  int edgeLayerShift = (Ordering == 0) ? 1 : numLayers;

  this->SetupFieldData(comm, neq_, req, sis, worksetSize);

  metaData->commit();

  bulkData->modification_begin(); // Begin modifying the mesh

  stk_classic::mesh::PartVector nodePartVec;
  stk_classic::mesh::PartVector singlePartVec(1);
  stk_classic::mesh::PartVector emptyPartVec;
  unsigned int ebNo = 0; //element block #???

  singlePartVec[0] = nsPartVec["Bottom"];

  AbstractSTKFieldContainer::IntScalarFieldType* proc_rank_field = fieldContainer->getProcRankField();
  AbstractSTKFieldContainer::VectorFieldType* coordinates_field = fieldContainer->getCoordinatesField();

  std::vector<long long int> prismMpasIds(NumBaseElemeNodes), prismGlobalIds(2 * NumBaseElemeNodes);

  for (int i = 0; i < (numLayers + 1) * nodes.size(); i++) {
    int ib = (Ordering == 0) * (i % lVertexColumnShift) + (Ordering == 1) * (i / vertexLayerShift);
    int il = (Ordering == 0) * (i / lVertexColumnShift) + (Ordering == 1) * (i % vertexLayerShift);
    stk_classic::mesh::Entity* node;
    stk_classic::mesh::Entity* node2d = nodes[ib];
    stk_classic::mesh::EntityId node2dId = node2d->identifier() - 1;
    if (il == 0)
      node = &bulkData->declare_entity(metaData->node_rank(), il * vertexColumnShift + vertexLayerShift * node2dId + 1, singlePartVec);
    else
      node = &bulkData->declare_entity(metaData->node_rank(), il * vertexColumnShift + vertexLayerShift * node2dId + 1, nodePartVec);

    double* coord = stk_classic::mesh::field_data(*coordinates_field, *node);
    double* coord2d = stk_classic::mesh::field_data(*coordinates_field, *node2d);
    coord[0] = coord2d[0];
    coord[1] = coord2d[1];

    int lid = nodes_map.LID((long long int)(node2dId));
    coord[2] = (*sHeightVec)[lid] - (*thickVec)[lid] * (1. - levelsNormalizedThickness[il]);

    if(hasSurfaceHeight) {
      double* sHeight = stk_classic::mesh::field_data(*fieldContainer->getSurfaceHeightField(), *node);
      sHeight[0] = (*sHeightVec)[lid];
    }

    if(hasThickness) {
      double* thick = stk_classic::mesh::field_data(*fieldContainer->getThicknessField(), *node);
      thick[0] = (*thickVec)[lid];
    }

    if(hasSurfaceVelocity) {
      double* sVelocity = stk_classic::mesh::field_data(*fieldContainer->getSurfaceVelocityField(), *node);
      sVelocity[0] = (*sVelocityVec)[0][lid];
      sVelocity[1] = (*sVelocityVec)[1][lid];
    }

    if(hasVelocityRMS) {
      double* velocityRMS = stk_classic::mesh::field_data(*fieldContainer->getVelocityRMSField(), *node);
      velocityRMS[0] = (*velocityRMSVec)[0][lid];
      velocityRMS[1] = (*velocityRMSVec)[1][lid];
    }

    if(hasBasalFriction) {
      double* bFriction = stk_classic::mesh::field_data(*fieldContainer->getBasalFrictionField(), *node);
      bFriction[0] = (*bFrictionVec)[lid];
    }

  }

  long long int tetrasLocalIdsOnPrism[3][4];

  for (int i = 0; i < cells.size() * numLayers; i++) {

    int ib = (Ordering == 0) * (i % lElemColumnShift) + (Ordering == 1) * (i / elemLayerShift);
    int il = (Ordering == 0) * (i / lElemColumnShift) + (Ordering == 1) * (i % elemLayerShift);

    int shift = il * vertexColumnShift;

    singlePartVec[0] = partVec[ebNo];

    //TODO: this could be done only in the first layer and then copied into the other layers

    stk_classic::mesh::PairIterRelation rel = cells[ib]->relations(meshStruct2D->metaData->node_rank());
    double tempOnPrism = 0; //Set temperature constant on each prism/Hexa
    for (int j = 0; j < NumBaseElemeNodes; j++) {
      stk_classic::mesh::EntityId node2dId = rel[j].entity()->identifier() - 1;
      int node2dLId = nodes_map.LID((long long int)(node2dId));
      stk_classic::mesh::EntityId mpasLowerId = vertexLayerShift * node2dId;
      stk_classic::mesh::EntityId lowerId = shift + vertexLayerShift * node2dId;
      prismMpasIds[j] = mpasLowerId;
      prismGlobalIds[j] = lowerId;
      prismGlobalIds[j + NumBaseElemeNodes] = lowerId + vertexColumnShift;
      if(hasTemperature)
        tempOnPrism += 1. / NumBaseElemeNodes / 2. * ((*temperatureVecInterp)[il][node2dLId] + (*temperatureVecInterp)[il + 1][node2dLId]);
    }

    switch (ElemShape) {
    case Tetrahedron: {
      tetrasFromPrismStructured(&prismMpasIds[0], &prismGlobalIds[0], tetrasLocalIdsOnPrism);

      stk_classic::mesh::EntityId prismId = il * elemColumnShift + elemLayerShift * (cells[ib]->identifier() - 1);
      for (int iTetra = 0; iTetra < 3; iTetra++) {
        stk_classic::mesh::Entity& elem = bulkData->declare_entity(metaData->element_rank(), 3 * prismId + iTetra + 1, singlePartVec);
        for (int j = 0; j < 4; j++) {
          stk_classic::mesh::Entity& node = *bulkData->get_entity(metaData->node_rank(), tetrasLocalIdsOnPrism[iTetra][j] + 1);
          bulkData->declare_relation(elem, node, j);
        }
        int* p_rank = (int*) stk_classic::mesh::field_data(*proc_rank_field, elem);
        p_rank[0] = comm->MyPID();
        if(hasTemperature) {
          double* temperature = stk_classic::mesh::field_data(*fieldContainer->getTemperatureField(), elem);
          temperature[0] = tempOnPrism;
        }
      }
    }
      break;
    case Wedge:
    case Hexahedron: {
      stk_classic::mesh::EntityId prismId = il * elemColumnShift + elemLayerShift * (cells[ib]->identifier() - 1);
      stk_classic::mesh::Entity& elem = bulkData->declare_entity(metaData->element_rank(), prismId + 1, singlePartVec);
      for (int j = 0; j < 2 * NumBaseElemeNodes; j++) {
        stk_classic::mesh::Entity& node = *bulkData->get_entity(metaData->node_rank(), prismGlobalIds[j] + 1);
        bulkData->declare_relation(elem, node, j);
      }
      int* p_rank = (int*) stk_classic::mesh::field_data(*proc_rank_field, elem);
      p_rank[0] = comm->MyPID();
      if(hasTemperature) {
        double* temperature = stk_classic::mesh::field_data(*fieldContainer->getTemperatureField(), elem);
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

  for (int i = 0; i < edges.size() * numLayers; i++) {
    int ib = (Ordering == 0) * (i % lEdgeColumnShift) + (Ordering == 1) * (i / edgeLayerShift);
    // if(!isBoundaryEdge[ib]) continue; //WARNING: assuming that all the edges stored are boundary edges!!

    stk_classic::mesh::Entity* edge2d = edges[ib];
    stk_classic::mesh::PairIterRelation rel = edge2d->relations(meshStruct2D->metaData->element_rank());
    int il = (Ordering == 0) * (i / lEdgeColumnShift) + (Ordering == 1) * (i % edgeLayerShift);
    stk_classic::mesh::Entity* elem2d = rel[0].entity();
    stk_classic::mesh::EntityId edgeLID = rel[0].identifier();

    stk_classic::mesh::EntityId basalElemId = elem2d->identifier() - 1;
    stk_classic::mesh::EntityId Edge2dId = edge2d->identifier() - 1;
    switch (ElemShape) {
    case Tetrahedron: {
      rel = elem2d->relations(meshStruct2D->metaData->node_rank());
      for (int j = 0; j < NumBaseElemeNodes; j++) {
        stk_classic::mesh::EntityId node2dId = rel[j].entity()->identifier() - 1;
        prismMpasIds[j] = vertexLayerShift * node2dId;
      }
      int minIndex;
      int pType = prismType(&prismMpasIds[0], minIndex);
      stk_classic::mesh::EntityId tetraId = 3 * il * elemColumnShift + 3 * elemLayerShift * basalElemId;

      stk_classic::mesh::Entity& elem0 = *bulkData->get_entity(metaData->element_rank(), tetraId + tetraAdjacentToPrismLateralFace[minIndex][pType][edgeLID][0] + 1);
      stk_classic::mesh::Entity& elem1 = *bulkData->get_entity(metaData->element_rank(), tetraId + tetraAdjacentToPrismLateralFace[minIndex][pType][edgeLID][1] + 1);

      stk_classic::mesh::Entity& side0 = bulkData->declare_entity(metaData->side_rank(), 2 * edgeColumnShift * il +  2 * Edge2dId * edgeLayerShift + 1, singlePartVec);
      stk_classic::mesh::Entity& side1 = bulkData->declare_entity(metaData->side_rank(), 2 * edgeColumnShift * il +  2 * Edge2dId * edgeLayerShift + 1 + 1, singlePartVec);

      bulkData->declare_relation(elem0, side0, tetraFaceIdOnPrismFaceId[minIndex][edgeLID]);
      bulkData->declare_relation(elem1, side1, tetraFaceIdOnPrismFaceId[minIndex][edgeLID]);

      stk_classic::mesh::PairIterRelation rel_elemNodes0 = elem0.relations(metaData->node_rank());
      stk_classic::mesh::PairIterRelation rel_elemNodes1 = elem1.relations(metaData->node_rank());
      for (int j = 0; j < 3; j++) {
        stk_classic::mesh::Entity& node0 = *rel_elemNodes0[this->meshSpecs[0]->ctd.side[tetraFaceIdOnPrismFaceId[minIndex][edgeLID]].node[j]].entity();
        bulkData->declare_relation(side0, node0, j);
        stk_classic::mesh::Entity& node1 = *rel_elemNodes1[this->meshSpecs[0]->ctd.side[tetraFaceIdOnPrismFaceId[minIndex][edgeLID]].node[j]].entity();
        bulkData->declare_relation(side1, node1, j);
      }
    }

      break;
    case Wedge:
    case Hexahedron: {
      stk_classic::mesh::EntityId prismId = il * elemColumnShift + elemLayerShift * basalElemId;
      stk_classic::mesh::Entity& elem = *bulkData->get_entity(metaData->element_rank(), prismId + 1);
      stk_classic::mesh::Entity& side = bulkData->declare_entity(metaData->side_rank(), edgeColumnShift * il +Edge2dId * edgeLayerShift + 1, singlePartVec);
      bulkData->declare_relation(elem, side, edgeLID);

      stk_classic::mesh::PairIterRelation rel_elemNodes = elem.relations(metaData->node_rank());
      for (int j = 0; j < 4; j++) {
        stk_classic::mesh::Entity& node = *rel_elemNodes[this->meshSpecs[0]->ctd.side[edgeLID].node[j]].entity();
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


  long long int edgeOffset = maxGlobalEdges2D * numLayers;
  if(ElemShape == Tetrahedron) edgeOffset *= 2;

  for (int i = 0; i < cells.size(); i++) {
    stk_classic::mesh::Entity& elem2d = *cells[i];
    stk_classic::mesh::EntityId elem2d_id = elem2d.identifier() - 1;
    stk_classic::mesh::Entity& side = bulkData->declare_entity(metaData->side_rank(), elem2d_id + edgeOffset + 1, singlePartVec);
    stk_classic::mesh::Entity& elem = *bulkData->get_entity(metaData->element_rank(), elem2d_id * numSubelemOnPrism * elemLayerShift + 1);
    bulkData->declare_relation(elem, side, basalSideLID);

    stk_classic::mesh::PairIterRelation rel_elemNodes = elem.relations(metaData->node_rank());
    for (int j = 0; j < numBasalSidePoints; j++) {
      stk_classic::mesh::Entity& node = *rel_elemNodes[this->meshSpecs[0]->ctd.side[basalSideLID].node[j]].entity();
      bulkData->declare_relation(side, node, j);
    }
  }

  singlePartVec[0] = ssPartVec["upperside"];

  edgeOffset += maxGlobalElements2D;

  for (int i = 0; i < cells.size(); i++) {
    stk_classic::mesh::Entity& elem2d = *cells[i];
    stk_classic::mesh::EntityId elem2d_id = elem2d.identifier() - 1;
    stk_classic::mesh::Entity& side = bulkData->declare_entity(metaData->side_rank(), elem2d_id  + edgeOffset + 1, singlePartVec);
    stk_classic::mesh::Entity& elem = *bulkData->get_entity(metaData->element_rank(), elem2d_id * numSubelemOnPrism * elemLayerShift + (numLayers - 1) * numSubelemOnPrism * elemColumnShift + 1 + (numSubelemOnPrism - 1));
    bulkData->declare_relation(elem, side, upperSideLID);

    stk_classic::mesh::PairIterRelation rel_elemNodes = elem.relations(metaData->node_rank());
    for (int j = 0; j < numBasalSidePoints; j++) {
      stk_classic::mesh::Entity& node = *rel_elemNodes[this->meshSpecs[0]->ctd.side[upperSideLID].node[j]].entity();
      bulkData->declare_relation(side, node, j);
    }
  }

  bulkData->modification_end();

}

Teuchos::RCP<const Teuchos::ParameterList> Albany::ExtrudedSTKMeshStruct::getValidDiscretizationParameters() const {
  Teuchos::RCP<Teuchos::ParameterList> validPL = this->getValidGenericSTKParameters("Valid Extruded_DiscParams");
  validPL->set<std::string>("Exodus Input File Name", "", "File Name For Exodus Mesh Input");
  validPL->set<std::string>("Surface Height File Name", "surface_height.ascii", "Name of the file containing the surface height data");
  validPL->set<std::string>("Thickness File Name", "thickness.ascii", "Name of the file containing the thickness data");
  validPL->set<std::string>("Surface Velocity File Name", "surface_velocity.ascii", "Name of the file containing the surface velocity data");
  validPL->set<std::string>("Velocity RMS File Name", "velocity_RMS.ascii", "Name of the file containing the surface velocity RMS data");
  validPL->set<std::string>("Basal Friction File Name", "basal_friction.ascii", "Name of the file containing the basal friction data");
  validPL->set<std::string>("Temperature File Name", "temperature.ascii", "Name of the file containing the temperature data");
  validPL->set<std::string>("Element Shape", "Hexahedron", "Shape of the Element: Tetrahedron, Wedge, Hexahedron");
  validPL->set<int>("NumLayers", 10, "Number of vertical Layers of the extruded mesh. In a vertical column, the mesh will have numLayers+1 nodes");
  validPL->set<bool>("Use Glimmer Spacing", false, "When true, the layer spacing is computed according to Glimmer formula (layers are denser close to the bedrock)");
  validPL->set<bool>("Columnwise Ordering", false, "True for Columnwise ordering, false for Layerwise ordering");
  return validPL;
}

void Albany::ExtrudedSTKMeshStruct::read2DFileSerial(std::string &fname, Epetra_Vector& content, const Teuchos::RCP<const Epetra_Comm>& comm) {
  long long int numNodes;
  if (comm->MyPID() == 0) {
    std::ifstream ifile;
    ifile.open(fname.c_str());
    if (ifile.is_open()) {
      ifile >> numNodes;
      TEUCHOS_TEST_FOR_EXCEPTION(numNodes != content.MyLength(), Teuchos::Exceptions::InvalidParameterValue, std::endl << "Error in ExtrudedSTKMeshStruct: Number of nodes in file " << fname << " (" << numNodes << ") is different from the number expected (" << content.MyLength() << ")" << std::endl);

      for (long long int i = 0; i < numNodes; i++)
        ifile >> content[i];
      ifile.close();
    } else {
      std::cout << "Warning in ExtrudedSTKMeshStruct: Unable to open the file " << fname << std::endl;
    }
  }
}

void Albany::ExtrudedSTKMeshStruct::readFileSerial(std::string &fname, std::vector<Epetra_Vector>& contentVec, const Teuchos::RCP<const Epetra_Comm>& comm) {
  long long int numNodes, numComponents;
  if (comm->MyPID() == 0) {
    std::ifstream ifile;
    ifile.open(fname.c_str());
    if (ifile.is_open()) {
      ifile >> numNodes >> numComponents;
      TEUCHOS_TEST_FOR_EXCEPTION(numNodes != contentVec[0].MyLength(), Teuchos::Exceptions::InvalidParameterValue,
          std::endl << "Error in ExtrudedSTKMeshStruct: Number of nodes in file " << fname << " (" << numNodes << ") is different from the number expected (" << contentVec[0].MyLength() << ")" << std::endl);
      TEUCHOS_TEST_FOR_EXCEPTION(numComponents != contentVec.size(), Teuchos::Exceptions::InvalidParameterValue,
          std::endl << "Error in ExtrudedSTKMeshStruct: Number of components in file " << fname << " (" << numComponents << ") is different from the number expected (" << contentVec.size() << ")" << std::endl);
      for (int il = 0; il < numComponents; ++il)
        for (long long int i = 0; i < numNodes; i++)
          ifile >> contentVec[il][i];
      ifile.close();
    } else {
      std::cout << "Warning in ExtrudedSTKMeshStruct: Unable to open input file " << fname << std::endl;
      //	TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
      //			std::endl << "Error in ExtrudedSTKMeshStruct: Unable to open input file " << fname << std::endl);
    }
  }
}

void Albany::ExtrudedSTKMeshStruct::readFileSerial(std::string &fname, const Epetra_Map& map_serial, const Epetra_Map& map, const Epetra_Import& importOperator, std::vector<Epetra_Vector>& temperatureVec, std::vector<double>& zCoords, const Teuchos::RCP<const Epetra_Comm>& comm) {
  long long int numNodes;
  int numComponents;
  std::ifstream ifile;
  if (comm->MyPID() == 0) {
    ifile.open(fname.c_str());
    if (ifile.is_open()) {
      ifile >> numNodes >> numComponents;

    //  std::cout << "numNodes >> numComponents: " << numNodes << " " << numComponents << std::endl;

      TEUCHOS_TEST_FOR_EXCEPTION(numNodes != map_serial.NumMyElements(), Teuchos::Exceptions::InvalidParameterValue,
          std::endl << "Error in ExtrudedSTKMeshStruct: Number of nodes in file " << fname << " (" << numNodes << ") is different from the number expected (" << map_serial.NumMyElements() << ")" << std::endl);
    } else {
      std::cout << "Warning in ExtrudedSTKMeshStruct: Unable to open input file " << fname << std::endl;
      //	TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
      //			std::endl << "Error in ExtrudedSTKMeshStruct: Unable to open input file " << fname << std::endl);
    }
  }
  comm->Broadcast(&numComponents, 1, 0);
  zCoords.resize(numComponents);
  Epetra_Vector tempT(map_serial);

  if (comm->MyPID() == 0) {
    for (int i = 0; i < numComponents; ++i)
      ifile >> zCoords[i];
  }
  comm->Broadcast(&zCoords[0], numComponents, 0);

  temperatureVec.resize(numComponents, Epetra_Vector(map));

  for (int il = 0; il < numComponents; ++il) {
    if (comm->MyPID() == 0)
      for (long long int i = 0; i < numNodes; i++)
        ifile >> tempT[i];
    temperatureVec[il].Import(tempT, importOperator, Insert);
  }

  if (comm->MyPID() == 0)
    ifile.close();

}

