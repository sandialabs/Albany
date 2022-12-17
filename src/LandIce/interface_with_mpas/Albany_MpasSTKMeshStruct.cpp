//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_MpasSTKMeshStruct.hpp"
#include <Albany_STKNodeSharing.hpp>
#include "Albany_Utils.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

#include "Teuchos_VerboseObject.hpp"

#include <Shards_BasicTopologies.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Selector.hpp>

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#include <Ioss_SubSystem.h>
#endif

#include <iostream>

namespace Albany
{

MpasSTKMeshStruct::
MpasSTKMeshStruct(const Teuchos::RCP<Teuchos::ParameterList>& params,
    const Teuchos::RCP<const Teuchos_Comm>& comm,
    const std::vector<int>& indexToVertexID_,
    const std::vector<int>& vertexProcIDs_,
    const std::vector<double>& verticesCoords_,
    int globalVerticesStride_,
    const std::vector<int>& verticesOnTria_,
    const std::vector<std::vector<int>>&  procsSharingVertices_,
    const std::vector<bool>& isBoundaryEdge_,
    const std::vector<int>& trianglesOnEdge_,
    const std::vector<int>& verticesOnEdge_,
    const std::vector<int>& indexToEdgeID_,
    int globalEdgesStride_,
    const std::vector<GO>& indexToTriangleID_,
    int globalTrianglesStride_,
    const std::vector<int>& dirichletNodesIds_,
    const std::vector<int>& iceMarginEdgesIds_,
    int numLayers_, const int numParams, int ordering) :
    GenericSTKMeshStruct(params, 3, numParams),
    out(Teuchos::VerboseObjectBase::getDefaultOStream()),
    periodic(false),
    NumEles(indexToTriangleID_.size()),
    hasRestartSol(false),
    restartTime(0.),
    indexToVertexID(indexToVertexID_),
    vertexProcIDs(vertexProcIDs_),
    verticesCoords(verticesCoords_),
    globalVerticesStride(globalVerticesStride_),
    verticesOnTria(verticesOnTria_),
    procsSharingVertices(procsSharingVertices_),
    isBoundaryEdge(isBoundaryEdge_),
    trianglesOnEdge(trianglesOnEdge_),
    verticesOnEdge(verticesOnEdge_),
    indexToEdgeID(indexToEdgeID_),
    globalEdgesStride(globalEdgesStride_),
    indexToTriangleID(indexToTriangleID_),
    globalTrianglesStride(globalTrianglesStride_),
    dirichletNodesIds(dirichletNodesIds_),
    iceMarginEdgesIds(iceMarginEdgesIds_),
    numLayers(numLayers_)
{
  auto LAYER  = LayeredMeshOrdering::LAYER;
  auto COLUMN = LayeredMeshOrdering::COLUMN;

  Ordering = (ordering==0) ? LAYER : COLUMN;

  std::string shape = params->get<std::string>("Element Shape");
  std::string basalside_elem_name;
  if(shape == "Tetrahedron")  {
    ElemShape = Tetrahedron;
  }
  else if (shape == "Prism")  {
    ElemShape = Wedge;
  }
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameterValue,
        std::endl << "Error in MpasSTKMeshStruct: Element Shape " << shape << " not recognized. Possible values: Tetrahedron, Prism");

  int numElemsInPrism = (ElemShape==Tetrahedron) ? 3 : 1;

  std::vector<GO> indexToElemID(numElemsInPrism*indexToTriangleID.size()*numLayers);

  int elemColumnShift = (Ordering == COLUMN) ? numElemsInPrism : numElemsInPrism*globalTrianglesStride;
  int lElemColumnShift = (Ordering == COLUMN) ? numElemsInPrism : numElemsInPrism*indexToTriangleID.size();
  int elemLayerShift = (Ordering == LAYER) ? numElemsInPrism : numElemsInPrism*numLayers;

  for(int il=0; il< numLayers; ++il) {
    int shift = il*elemColumnShift;
    int lShift = il*lElemColumnShift;
    for(int j=0; j<static_cast<int>(indexToTriangleID.size()); ++j) {
      for(int iElem=0; iElem < numElemsInPrism; ++iElem) {
        int lid = lShift + j*elemLayerShift + iElem;
        indexToElemID[lid] = shift+elemLayerShift * (indexToTriangleID[j]-1) + iElem;
      }
    }
  }

  auto indexToElemIDAV = Teuchos::arrayViewFromVector(indexToElemID);
  // Distribute the elems equally. Build total_elems elements, with nodeIDs starting at StartIndex
  int nLocalTriangles = indexToTriangleID.size(), nGlobalTriangles;
  Teuchos::reduceAll<int, int> (*comm, Teuchos::REDUCE_SUM, 1, &nLocalTriangles, &nGlobalTriangles);
  elem_vs = createVectorSpace(comm,indexToElemIDAV,GO(numElemsInPrism*nGlobalTriangles*numLayers));

  params->validateParameters(*getValidDiscretizationParameters(),0);

  std::vector<std::string> nsNames;
  std::string nsn="lateral";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, stk::topology::NODE_RANK );
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn="internal";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, stk::topology::NODE_RANK );
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn="bottom";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, stk::topology::NODE_RANK );
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn="dirichlet";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, stk::topology::NODE_RANK );
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif

  std::vector<std::string> ssNames;
  std::string ssnLat="lateralside";
  std::string ssnBottom="basalside";
  std::string ssnTop="upperside";
  std::string ssnLatFloat="ice_margin_side";

  ssNames.push_back(ssnLat);
  ssNames.push_back(ssnBottom);
  ssNames.push_back(ssnTop);
  ssNames.push_back(ssnLatFloat);
  ssPartVec[ssnLat] = & metaData->declare_part(ssnLat, metaData->side_rank() );
  ssPartVec[ssnBottom] = & metaData->declare_part(ssnBottom, metaData->side_rank() );
  ssPartVec[ssnTop] = & metaData->declare_part(ssnTop, metaData->side_rank() );
  ssPartVec[ssnLatFloat] = & metaData->declare_part(ssnLatFloat, metaData->side_rank() );
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*ssPartVec[ssnLat]);
  stk::io::put_io_part_attribute(*ssPartVec[ssnBottom]);
  stk::io::put_io_part_attribute(*ssPartVec[ssnTop]);
  stk::io::put_io_part_attribute(*ssPartVec[ssnLatFloat]);
#endif

  stk::topology etopology;

  switch (ElemShape) {
  case Tetrahedron:
    etopology = stk::topology::TET_4;
    stk::mesh::set_topology(*ssPartVec[ssnBottom],stk::topology::TRI_3);
    stk::mesh::set_topology(*ssPartVec[ssnTop],stk::topology::TRI_3);
    stk::mesh::set_topology(*ssPartVec[ssnLat],stk::topology::TRI_3);
    stk::mesh::set_topology(*ssPartVec[ssnLatFloat],stk::topology::TRI_3);
    break;
  case Wedge:
    etopology = stk::topology::WEDGE_6;
    stk::mesh::set_topology(*ssPartVec[ssnBottom],stk::topology::TRI_3);
    stk::mesh::set_topology(*ssPartVec[ssnTop],stk::topology::TRI_3);
    stk::mesh::set_topology(*ssPartVec[ssnLat],stk::topology::QUAD_4);
    stk::mesh::set_topology(*ssPartVec[ssnLatFloat],stk::topology::QUAD_4);
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
  int cub = params->get("Cubature Degree",3);
  int worksetSizeMax = params->get("Workset Size",50);
  int worksetSize = this->computeWorksetSize(worksetSizeMax, elem_vs->localSubDim());

  const CellTopologyData& ctd = *shards_ctd.getCellTopologyData();

  this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub, nsNames, ssNames, worksetSize,
     ebn, ebNameToIndex, this->interleavedOrdering));

  this->initializeSideSetMeshSpecs(comm);
  this->initializeSideSetMeshStructs(comm);
}

void MpasSTKMeshStruct::setFieldData(
              const Teuchos::RCP<const Teuchos_Comm>& comm,
              const AbstractFieldContainer::FieldContainerRequirements& /*req*/,
              const Teuchos::RCP<StateInfoStruct>& sis,
              const unsigned int worksetSize,
              const std::map<std::string,Teuchos::RCP<StateInfoStruct> >& side_set_sis,
              const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& /*side_set_req*/)
{
  Albany::AbstractFieldContainer::FieldContainerRequirements req;
  this->SetupFieldData(comm, req, sis, worksetSize);
  this->setSideSetFieldData(comm, {}, side_set_sis, worksetSize);
}

void MpasSTKMeshStruct::setBulkData(
    const Teuchos::RCP<const Teuchos_Comm>& comm,
    const Albany::AbstractFieldContainer::FieldContainerRequirements& /*req*/,
    const Teuchos::RCP<Albany::StateInfoStruct>& /* sis */,
    const unsigned int worksetSize,
    const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis,
    const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& /*side_set_req*/)
{
  int numElemsInPrism = (ElemShape==Tetrahedron) ? 3 : 1;
  int numLatSidesInQuad = (ElemShape==Tetrahedron) ? 2 : 1;

  auto LAYER  = LayeredMeshOrdering::LAYER;
  auto COLUMN = LayeredMeshOrdering::COLUMN;

  int elemColumnShift = (Ordering == COLUMN) ? numElemsInPrism : numElemsInPrism*globalTrianglesStride;
  int lElemColumnShift = (Ordering == COLUMN) ? numElemsInPrism : numElemsInPrism*indexToTriangleID.size();
  int elemLayerShift = (Ordering == LAYER) ? numElemsInPrism : numElemsInPrism*numLayers;

  GO vertexColumnShift = (Ordering == COLUMN) ? 1 : globalVerticesStride;
  int lVertexColumnShift = (Ordering == COLUMN) ? 1 : indexToVertexID.size();
  int vertexLayerShift = (Ordering == LAYER) ? 1 : numLayers+1;

  int edgeColumnShift = (Ordering == COLUMN) ? numLatSidesInQuad : numLatSidesInQuad*globalEdgesStride;
  int lEdgeColumnShift = (Ordering == COLUMN) ? 1 : indexToEdgeID.size();
  int edgeLayerShift = (Ordering == LAYER) ? 1 : numLayers;

  Teuchos::ArrayRCP<double> layerThicknessRatio(numLayers, 1.0/double(numLayers));
  this->global_cell_layers_data = (Ordering==LAYER) ?
      Teuchos::rcp(new LayeredMeshNumbering<GO>(elemColumnShift,Ordering,layerThicknessRatio)):
      Teuchos::rcp(new LayeredMeshNumbering<GO>(static_cast<GO>(elemLayerShift),Ordering,layerThicknessRatio));

  this->local_cell_layers_data = (Ordering==LAYER) ?
      Teuchos::rcp(new LayeredMeshNumbering<LO>(lElemColumnShift,Ordering,layerThicknessRatio)):
      Teuchos::rcp(new LayeredMeshNumbering<LO>(elemLayerShift,Ordering,layerThicknessRatio));


  metaData->commit();

  bulkData->modification_begin(); // Begin modifying the mesh

  stk::mesh::PartVector nodePartVec;
  stk::mesh::PartVector singlePartVec(1);
  stk::mesh::PartVector emptyPartVec;
  std::cout << "elem_map # elements: " << elem_vs->localSubDim() << std::endl;
  unsigned int ebNo = 0; //element block #???

  singlePartVec[0] = nsPartVec["bottom"];

  AbstractSTKFieldContainer::IntScalarFieldType* proc_rank_field = fieldContainer->getProcRankField();
  AbstractSTKFieldContainer::VectorFieldType* coordinates_field = fieldContainer->getCoordinatesField();

  for(int i=0; i< (numLayers+1)*static_cast<int>(indexToVertexID.size()); i++) {
    int ib = (Ordering == LAYER)*(i%lVertexColumnShift) + (Ordering == COLUMN)*(i/vertexLayerShift);
    int il = (Ordering == LAYER)*(i/lVertexColumnShift) + (Ordering == COLUMN)*(i%vertexLayerShift);

    stk::mesh::Entity node;
    if(il == 0) {
      node = bulkData->declare_entity(stk::topology::NODE_RANK, il*vertexColumnShift+vertexLayerShift*(indexToVertexID[ib]-1)+1, singlePartVec);
    } else {
      node = bulkData->declare_entity(stk::topology::NODE_RANK, il*vertexColumnShift+vertexLayerShift*(indexToVertexID[ib]-1)+1, nodePartVec);
    }

    auto sharing_procs = procsSharingVertices[ib];
    for(int iproc=0; iproc<static_cast<int>(sharing_procs.size()); ++iproc) {
      bulkData->add_node_sharing(node, sharing_procs[iproc]);
    }

    double* coord = stk::mesh::field_data(*coordinates_field, node);
    coord[0] = verticesCoords[3*ib];   coord[1] = verticesCoords[3*ib+1]; coord[2] = double(il)/numLayers;
  }

  singlePartVec[0] = nsPartVec["dirichlet"];
  for(int i=0; i<static_cast<int>(dirichletNodesIds.size()); ++i) {
    stk::mesh::Entity node = bulkData->get_entity(stk::topology::NODE_RANK, dirichletNodesIds[i]);
    bulkData->change_entity_parts(node, singlePartVec);
  }

  int tetrasLocalIdsOnPrism[3][4];
  auto elem_vs_indexer = Albany::createGlobalLocalIndexer(elem_vs);
  for (unsigned int i=0; i<elem_vs->localSubDim()/numElemsInPrism; i++) {
    int ib = (Ordering == LAYER)*(i%(lElemColumnShift/numElemsInPrism)) + (Ordering == COLUMN)*(i/(elemLayerShift/numElemsInPrism));
    int il = (Ordering == LAYER)*(i/(lElemColumnShift/numElemsInPrism)) + (Ordering == COLUMN)*(i%(elemLayerShift/numElemsInPrism));

    int shift = il*vertexColumnShift;

    singlePartVec[0] = partVec[ebNo];
    //TODO: this could be done only in the first layer and then copied into the other layers
    int prismGlobalIds[6];
    for (unsigned int j = 0; j < 3; ++j) {
      int lowerId = shift+vertexLayerShift*(indexToVertexID[verticesOnTria[3*ib+j]]-1);
      prismGlobalIds[j] = lowerId;
      prismGlobalIds[j + 3] = lowerId+vertexColumnShift;
    }


    switch (ElemShape)
    {
    case Tetrahedron:
    {
      tetrasFromPrismStructured(prismGlobalIds, tetrasLocalIdsOnPrism);

      for (unsigned int iTetra = 0; iTetra < 3; iTetra++) {
        stk::mesh::Entity elem = bulkData->declare_entity(stk::topology::ELEMENT_RANK, elem_vs_indexer->getGlobalElement(3*i+iTetra)+1, singlePartVec);
        for(int j=0; j<4; ++j) {
          stk::mesh::Entity node = bulkData->get_entity(stk::topology::NODE_RANK, tetrasLocalIdsOnPrism[iTetra][j]+1);
          bulkData->declare_relation(elem, node, j);
        }
        if(proc_rank_field){
          int* p_rank = (int*)stk::mesh::field_data(*proc_rank_field, elem);
          if(p_rank) 
			p_rank[0] = comm->getRank();
        }
      }
      break;
    }
    case Wedge:
    {
      stk::mesh::Entity elem = bulkData->declare_entity(stk::topology::ELEMENT_RANK, elem_vs_indexer->getGlobalElement(i) + 1, singlePartVec);
      for (unsigned int j = 0; j < 6; j++) {
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

  // we store the lower and upper faces of prisms, which corresponds to triangles of the basal mesh
  int maxLocalTriangleId(0), maxGlobalTriangleId;
  if( indexToTriangleID.size() > 0)
    maxLocalTriangleId = *std::max_element(indexToTriangleID.begin(), indexToTriangleID.end());
  Teuchos::reduceAll<int, int> (*comm, Teuchos::REDUCE_MAX, 1, &maxLocalTriangleId, &maxGlobalTriangleId);

  //position of side in element depends on how the tetrahedron is located in the Prism.
  int basalSideLID = 3;
  int upperSideLID = (ElemShape == Tetrahedron) ? 1 : 4;

  singlePartVec[0] = ssPartVec["basalside"];
  for (unsigned int i=0; i<indexToTriangleID.size(); ++i) {
    stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), indexToTriangleID[i], singlePartVec);
    stk::mesh::Entity elem  = bulkData->get_entity(stk::topology::ELEMENT_RANK,  (indexToTriangleID[i]-1)*elemLayerShift+1);
    bulkData->declare_relation(elem, side,  basalSideLID);
    stk::mesh::Entity const* rel_elemNodes = bulkData->begin_nodes(elem);
    for(int j=0; j<3; ++j) {
      stk::mesh::Entity node = rel_elemNodes[this->meshSpecs[0]->ctd.side[basalSideLID].node[j]];
      bulkData->declare_relation(side, node, j);
    }
  }

  int upperBasalOffset = maxGlobalTriangleId+1;

  singlePartVec[0] = ssPartVec["upperside"];
  for (unsigned int i=0; i<indexToTriangleID.size(); ++i) {
    stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), indexToTriangleID[i]+upperBasalOffset, singlePartVec);
    stk::mesh::Entity elem  = bulkData->get_entity(stk::topology::ELEMENT_RANK,  (indexToTriangleID[i]-1)*elemLayerShift+(numLayers-1)*elemColumnShift+1+(numElemsInPrism-1));
    bulkData->declare_relation(elem, side,  upperSideLID);
    stk::mesh::Entity const* rel_elemNodes = bulkData->begin_nodes(elem);
    for(int j=0; j<3; ++j) {
      stk::mesh::Entity node = rel_elemNodes[this->meshSpecs[0]->ctd.side[upperSideLID].node[j]];
      bulkData->declare_relation(side, node, j);
    }
  }

  upperBasalOffset += maxGlobalTriangleId+1;

  singlePartVec[0] = ssPartVec["lateralside"];
  //first we store the lateral faces of prisms, which corresponds to edges of the basal mesh
  int tetraSidePoints[4][3] = {{0, 1, 3}, {1, 2, 3}, {0, 3, 2}, {0, 2, 1}};
  std::vector<int> tetraPos(2), facePos(2);
  std::vector<std::vector<std::vector<int> > > prismStruct(3, std::vector<std::vector<int> >(4, std::vector<int>(3)));
  std::vector<int> bdPrismFaceIds(4);
  for (unsigned int i=0; i<indexToEdgeID.size()*numLayers; ++i) {
    int ib = (Ordering == LAYER)*(i%lEdgeColumnShift) + (Ordering == COLUMN)*(i/edgeLayerShift);
    if(isBoundaryEdge[ib]) {
      int il = (Ordering == LAYER)*(i/lEdgeColumnShift) + (Ordering == COLUMN)*(i%edgeLayerShift);
      int lBasalElemId = trianglesOnEdge[2*ib];
      int basalElemId = indexToTriangleID[lBasalElemId]-1;

      //TODO: this could be done only in the first layer and then copied into the other layers
      int prismGlobalIds[6];
      int shift = il*vertexColumnShift;
      for (unsigned int j = 0; j < 3; ++j) {
        int lowerId = shift+vertexLayerShift*(indexToVertexID[verticesOnTria[3*lBasalElemId+j]]-1);
        prismGlobalIds[j] = lowerId;
        prismGlobalIds[j + 3] = lowerId+vertexColumnShift;
      }

      bdPrismFaceIds[0] = (indexToVertexID[verticesOnEdge[2*ib]]-1)*vertexLayerShift+vertexColumnShift*il+1;
      bdPrismFaceIds[1] = (indexToVertexID[verticesOnEdge[2*ib+1]]-1)*vertexLayerShift+vertexColumnShift*il+1;
      bdPrismFaceIds[2] = bdPrismFaceIds[0]+vertexColumnShift;
      bdPrismFaceIds[3] = bdPrismFaceIds[1]+vertexColumnShift;

      int basalEdgeId = indexToEdgeID[ib]*edgeLayerShift;

      switch (ElemShape) {
      case Tetrahedron: {
        tetrasFromPrismStructured (prismGlobalIds, tetrasLocalIdsOnPrism);

        for(int iTetra = 0; iTetra<3; ++iTetra) {
          std::vector<std::vector<int> >& tetraStruct =prismStruct[iTetra];
          stk::mesh::EntityId tetraPoints[4];
          for(int j=0; j<4; ++j) {
            tetraPoints[j] = tetrasLocalIdsOnPrism[iTetra][j]+1;
          }
          for(int iFace=0; iFace<4; ++iFace) {
            std::vector<int>&  face = tetraStruct[iFace];
            for(int j=0; j<3; j++) {
              face[j] = tetraPoints[tetraSidePoints[iFace][j]];
            }
          }
        }

        setBdFacesOnPrism (prismStruct, bdPrismFaceIds, tetraPos, facePos);


        for(int k=0; k<static_cast<int>(tetraPos.size()); ++k) {
          int iTetra = tetraPos[k];
          int iFace = facePos[k];
          stk::mesh::Entity elem = bulkData->get_entity(stk::topology::ELEMENT_RANK, il*elemColumnShift+elemLayerShift * basalElemId +iTetra+1);
          int sideId = edgeColumnShift*il+numLatSidesInQuad*basalEdgeId+k+1 + upperBasalOffset;
          stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), sideId, singlePartVec);
          bulkData->declare_relation(elem, side,  iFace );
          stk::mesh::Entity const* rel_elemNodes = bulkData->begin_nodes(elem);
          for(int j=0; j<3; ++j) {
            stk::mesh::Entity node = rel_elemNodes[this->meshSpecs[0]->ctd.side[iFace].node[j]];
            bulkData->declare_relation(side, node, j);
          }
        }
      }
      break;
      case Wedge: {
        int faceId0 = prismGlobalIds[0]+1, faceId1 = prismGlobalIds[1]+1, faceId2 = prismGlobalIds[2]+1;
        int prismFaceLID = (((faceId0==bdPrismFaceIds[0])&&(faceId1==bdPrismFaceIds[1])) ||
                            ((faceId0==bdPrismFaceIds[1])&&(faceId1==bdPrismFaceIds[0]))) ? 0 :
                           (((faceId1==bdPrismFaceIds[0])&&(faceId2==bdPrismFaceIds[1])) ||
                            ((faceId1==bdPrismFaceIds[1])&&(faceId2==bdPrismFaceIds[0]))) ? 1 : 2;

        stk::mesh::EntityId sideId = edgeColumnShift*il+basalEdgeId+1 + upperBasalOffset;
        stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), sideId, singlePartVec);

        stk::mesh::EntityId prismId = il * elemColumnShift + elemLayerShift * basalElemId;
        stk::mesh::Entity elem = bulkData->get_entity(stk::topology::ELEMENT_RANK, prismId + 1);
        bulkData->declare_relation(elem, side, prismFaceLID);

        stk::mesh::Entity const* rel_elemNodes = bulkData->begin_nodes(elem);
        for (unsigned int j = 0; j < 4; j++) {
          stk::mesh::Entity node = rel_elemNodes[this->meshSpecs[0]->ctd.side[prismFaceLID].node[j]];
          bulkData->declare_relation(side, node, j);
        }
      }
      break;
      }
    }
  }

  singlePartVec[0] = ssPartVec["ice_margin_side"];
  for(int i=0; i<static_cast<int>(iceMarginEdgesIds.size()); ++i) {
    int basalEdgeId = iceMarginEdgesIds[i]*edgeLayerShift;
    for(int il=0; il<numLayers; ++il) {
      for(int k=0; k< numLatSidesInQuad; ++k) {
        int sideId = edgeColumnShift*il+numLatSidesInQuad*basalEdgeId+k+1 + upperBasalOffset;
        stk::mesh::Entity side = bulkData->get_entity(metaData->side_rank(), sideId);
        bulkData->change_entity_parts(side, singlePartVec);
      }
    }
  }

  bulkData->modification_end();

  //change ownership to nodes to reflect MPAS one
  stk::mesh::EntityProcVec node_to_proc;
  for(int i=0; i< (numLayers+1)*static_cast<int>(indexToVertexID.size()); i++) {
    int ib = (Ordering == LAYER)*(i%lVertexColumnShift) + (Ordering == COLUMN)*(i/vertexLayerShift);
    int il = (Ordering == LAYER)*(i/lVertexColumnShift) + (Ordering == COLUMN)*(i%vertexLayerShift);

    stk::mesh::Entity node = bulkData->get_entity(stk::topology::NODE_RANK, il*vertexColumnShift+vertexLayerShift*(indexToVertexID[ib]-1)+1);
    int procID = vertexProcIDs[ib];
    if(bulkData->bucket(node).owned() && (procID != bulkData->parallel_rank()))
      node_to_proc.push_back(std::make_pair(node, procID));
  }

  bulkData->change_entity_owner(node_to_proc);

  Albany::AbstractFieldContainer::FieldContainerRequirements req;
  this->loadRequiredInputFields (req,comm);

  this->setSideSetBulkData(comm, {}, side_set_sis, worksetSize);
}

Teuchos::RCP<const Teuchos::ParameterList>
MpasSTKMeshStruct::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
      this->getValidGenericSTKParameters("Valid MpasSTKMeshStructParams");
  validPL->set<std::string>("Element Shape", "Tetrahedron", "Shape of the Element: Tetrahedron, Prism");

  return validPL;
}


int
MpasSTKMeshStruct::prismType(int const* prismVertexIds, int& minIndex)
{
  int PrismVerticesMap[6][6] = {{0, 1, 2, 3, 4, 5}, {1, 2, 0, 4, 5, 3}, {2, 0, 1, 5, 3, 4}, {3, 5, 4, 0, 2, 1}, {4, 3, 5, 1, 0, 2}, {5, 4, 3, 2, 1, 0}};
  minIndex = std::min_element (prismVertexIds, prismVertexIds + 3) - prismVertexIds;

  int v1 (prismVertexIds[PrismVerticesMap[minIndex][1]]);
  int v2 (prismVertexIds[PrismVerticesMap[minIndex][2]]);

  return v1  > v2;
}

void
MpasSTKMeshStruct::tetrasFromPrismStructured (int const* prismVertexGIds, int tetrasIdsOnPrism[][4])
{
  int PrismVerticesMap[6][6] = {{0, 1, 2, 3, 4, 5}, {1, 2, 0, 4, 5, 3}, {2, 0, 1, 5, 3, 4}, {3, 5, 4, 0, 2, 1}, {4, 3, 5, 1, 0, 2}, {5, 4, 3, 2, 1, 0}};

  int tetraOfPrism[2][3][4] = {{{0, 1, 2, 5}, {0, 1, 5, 4}, {0, 4, 5, 3}}, {{0, 1, 2, 4}, {0, 4, 2, 5}, {0, 4, 5, 3}}};

  int minIndex;
  int prismType = this->prismType(prismVertexGIds, minIndex);

  GO reorderedPrismLIds[6];

  for (unsigned int ii = 0; ii < 6; ii++)
  {
    reorderedPrismLIds[ii] = prismVertexGIds[PrismVerticesMap[minIndex][ii]];
  }

  for (unsigned int iTetra = 0; iTetra < 3; iTetra++)
    for (unsigned int iVertex = 0; iVertex < 4; iVertex++)
    {
      tetrasIdsOnPrism[iTetra][iVertex] = reorderedPrismLIds[tetraOfPrism[prismType][iTetra][iVertex]];
    }
}

void
MpasSTKMeshStruct::setBdFacesOnPrism (const std::vector<std::vector<std::vector<int> > >& prismStruct, const std::vector<int>& prismFaceIds, std::vector<int>& tetraPos, std::vector<int>& facePos)
{
  unsigned int numTriaFaces = prismFaceIds.size() - 2;
  tetraPos.assign(numTriaFaces,-1);
  facePos.assign(numTriaFaces,-1);


  for (unsigned int iTetra (0), k (0); (iTetra < 3 && k < numTriaFaces); iTetra++)
  {
    bool found;
    for (int jFaceLocalId = 0; jFaceLocalId < 4; jFaceLocalId++ )
    {
      found = true;
      for (int ip (0); ip < 3 && found; ip++)
      {
        int localId = prismStruct[iTetra][jFaceLocalId][ip];
        decltype(prismFaceIds.size()) j = 0;
        found = false;
        while ( (j < prismFaceIds.size()) && !found )
        {
          found = (localId == prismFaceIds[j]);
          j++;
        }
      }
      if (found)
      {
        tetraPos[k] = iTetra;
        facePos[k] = jFaceLocalId;
        k += found;
        break;
      }
    }
  }
}

} // namespace Albany
