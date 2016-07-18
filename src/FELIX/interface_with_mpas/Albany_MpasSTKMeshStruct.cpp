//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//9/12/14: Not compiled when ALBANY_EPETRA_EXE turned off.

#include <iostream>

#include "Albany_MpasSTKMeshStruct.hpp"
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

#include <boost/algorithm/string/predicate.hpp>

#include <Albany_STKNodeSharing.hpp>

#include "Albany_Utils.hpp"

//Wedge
Albany::MpasSTKMeshStruct::MpasSTKMeshStruct(const Teuchos::RCP<Teuchos::ParameterList>& params,
                                             const Teuchos::RCP<const Teuchos_Comm>& commT,
                                             const std::vector<GO>& indexToTriangleID, const std::vector<int>& verticesOnTria, int nGlobalTriangles, int numLayers, int ordering) :
  GenericSTKMeshStruct(params,Teuchos::null,3),
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  periodic(false),
  NumEles(indexToTriangleID.size()),
  hasRestartSol(false),
  restartTime(0.)
{
  auto LAYER  = LayeredMeshOrdering::LAYER;
  auto COLUMN = LayeredMeshOrdering::COLUMN;

  Ordering = (ordering==0) ? LAYER : COLUMN;

  std::vector<GO> indexToPrismID(indexToTriangleID.size()*numLayers);

  //Int ElemColumnShift = (ordering == ColumnWise) ? 1 : indexToTriangleID.size();
  int elemColumnShift = (Ordering == COLUMN) ? 1 : nGlobalTriangles;
  int lElemColumnShift = (Ordering == COLUMN) ? 1 : indexToTriangleID.size();
  int elemLayerShift = (Ordering == LAYER) ? 1 : numLayers;

  for(int il=0; il< numLayers; il++)
  {
	  int shift = il*elemColumnShift;
	  int lShift = il*lElemColumnShift;
	  for(int j=0; j< indexToTriangleID.size(); j++)
	  {
		  int lid = lShift + j*elemLayerShift;
		  indexToPrismID[lid] = shift+elemLayerShift * indexToTriangleID[j];
	  }
  }

  Teuchos::ArrayView<const GO> indexToPrismIDAV = Teuchos::arrayViewFromVector(indexToPrismID);

  // Distribute the elems equally. Build total_elems elements, with nodeIDs starting at StartIndex
  elem_mapT = Teuchos::rcp(new Tpetra_Map(nGlobalTriangles*numLayers, indexToPrismIDAV, 0, commT));

  params->validateParameters(*getValidDiscretizationParameters(),0);


  std::string ebn="Element Block 0";
  partVec[0] = & metaData->declare_part(ebn, stk::topology::ELEMENT_RANK );
  ebNameToIndex[ebn] = 0;

#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*partVec[0]);
#endif


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
  std::string ssnLatFloat="floatinglateralside";
  std::string ssnBottom="basalside";
  std::string ssnTop="upperside";

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

  stk::mesh::set_cell_topology<shards::Wedge<6> >(*partVec[0]);
  stk::mesh::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnBottom]);
  stk::mesh::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnTop]);
  stk::mesh::set_cell_topology<shards::Quadrilateral<4> >(*ssPartVec[ssnLat]);
  stk::mesh::set_cell_topology<shards::Quadrilateral<4> >(*ssPartVec[ssnLatFloat]);

  numDim = 3;
  int cub = params->get("Cubature Degree",3);
  int worksetSizeMax = params->get("Workset Size",50);
  int worksetSize = this->computeWorksetSize(worksetSizeMax, elem_mapT->getNodeNumElements());

  const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();

  this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub,
                             nsNames, ssNames, worksetSize, partVec[0]->name(),
                             ebNameToIndex, this->interleavedOrdering));
  //this->initializeSideSetMeshStructs(comm);


}

//Tetra
Albany::MpasSTKMeshStruct::MpasSTKMeshStruct(const Teuchos::RCP<Teuchos::ParameterList>& params,
                                             const Teuchos::RCP<const Teuchos_Comm>& commT,
                                             const std::vector<GO>& indexToTriangleID, int nGlobalTriangles, int numLayers, int ordering) :
  GenericSTKMeshStruct(params,Teuchos::null,3),
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  periodic(false),
  NumEles(indexToTriangleID.size()),
  hasRestartSol(false),
  restartTime(0.)
{
  auto LAYER  = LayeredMeshOrdering::LAYER;
  auto COLUMN = LayeredMeshOrdering::COLUMN;

  Ordering = (ordering==0) ? LAYER : COLUMN;

  std::vector<GO> indexToTetraID(3*indexToTriangleID.size()*numLayers);

  //Int ElemColumnShift = (ordering == ColumnWise) ? 1 : indexToTriangleID.size();
  int elemColumnShift = (Ordering == COLUMN) ? 3 : 3*nGlobalTriangles;
  int lElemColumnShift = (Ordering == COLUMN) ? 3 : 3*indexToTriangleID.size();
  int elemLayerShift = (Ordering == LAYER) ? 3 : 3*numLayers;

  for(int il=0; il< numLayers; il++)
  {
	  int shift = il*elemColumnShift;
	  int lShift = il*lElemColumnShift;
	  for(int j=0; j< indexToTriangleID.size(); j++)
	  {
		  for(int iTetra=0; iTetra<3; iTetra++)
		  {
			  int lid = lShift + j*elemLayerShift +iTetra;
			  indexToTetraID[lid] = shift+elemLayerShift * indexToTriangleID[j] +iTetra;
		  }
	  }
  }

  Teuchos::ArrayView<const GO> indexToTetraIDAV = Teuchos::arrayViewFromVector(indexToTetraID);
  // Distribute the elems equally. Build total_elems elements, with nodeIDs starting at StartIndex
  elem_mapT = Teuchos::rcp(new Tpetra_Map(3*nGlobalTriangles*numLayers, indexToTetraIDAV, 0, commT));

  params->validateParameters(*getValidDiscretizationParameters(),0);


  std::string ebn="Element Block 0";
  partVec[0] = & metaData->declare_part(ebn, stk::topology::ELEMENT_RANK );
  ebNameToIndex[ebn] = 0;

#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*partVec[0]);
#endif


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
  std::string ssnLatFloat="floatinglateralside";

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

  stk::mesh::set_cell_topology<shards::Tetrahedron<4> >(*partVec[0]);
  stk::mesh::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnBottom]);
  stk::mesh::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnTop]);
  stk::mesh::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnLat]);
  stk::mesh::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnLatFloat]);

  numDim = 3;
  int cub = params->get("Cubature Degree",3);
  int worksetSizeMax = params->get("Workset Size",50);
  int worksetSize = this->computeWorksetSize(worksetSizeMax, elem_mapT->getNodeNumElements());

  const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();

  this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub,
                             nsNames, ssNames, worksetSize, partVec[0]->name(),
                             ebNameToIndex, this->interleavedOrdering));


}


Albany::MpasSTKMeshStruct::~MpasSTKMeshStruct()
{
}

void
Albany::MpasSTKMeshStruct::constructMesh(
                                               const Teuchos::RCP<const Teuchos_Comm>& commT,
                                               const Teuchos::RCP<Teuchos::ParameterList>& params,
                                               const unsigned int neq_,
                                               const Albany::AbstractFieldContainer::FieldContainerRequirements& req,
                                               const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                                               const std::vector<int>& indexToVertexID, const std::vector<double>& verticesCoords, const std::vector<bool>& isVertexBoundary, int nGlobalVertices,
                                               const std::vector<int>& verticesOnTria,
                                               const std::vector<bool>& isBoundaryEdge, const std::vector<int>& trianglesOnEdge, const std::vector<int>& trianglesPositionsOnEdge,
                                               const std::vector<int>& verticesOnEdge,
                                               const std::vector<int>& indexToEdgeID, int nGlobalEdges,
                                               const std::vector<GO>& indexToTriangleID,
                                               const std::vector<int>& dirichletNodesIds,
                                               const std::vector<int>& floating2dLateralEdgesIds,
                                               const unsigned int worksetSize,
                                               int numLayers, int ordering)
{
	this->SetupFieldData(commT, neq_, req, sis, worksetSize);
  auto LAYER  = LayeredMeshOrdering::LAYER;
  auto COLUMN = LayeredMeshOrdering::COLUMN;

	Ordering = (ordering==0) ? LAYER : COLUMN;

  int elemColumnShift = (Ordering == COLUMN) ? 1 : elem_mapT->getGlobalNumElements()/numLayers;
  int lElemColumnShift = (Ordering == COLUMN) ? 1 : indexToTriangleID.size();
  int elemLayerShift = (Ordering == LAYER) ? 1 : numLayers;

  int vertexColumnShift = (Ordering == COLUMN) ? 1 : nGlobalVertices;
  int lVertexColumnShift = (Ordering == COLUMN) ? 1 : indexToVertexID.size();
  int vertexLayerShift = (Ordering == LAYER) ? 1 : numLayers+1;

  int edgeColumnShift = (Ordering == COLUMN) ? 1 : nGlobalEdges;
  int lEdgeColumnShift = (Ordering == COLUMN) ? 1 : indexToEdgeID.size();
  int edgeLayerShift = (Ordering == LAYER) ? 1 : numLayers;

  Teuchos::ArrayRCP<double> layerThicknessRatio(numLayers, 1.0/double(numLayers));
  this->layered_mesh_numbering = (Ordering == LAYER) ?
            Teuchos::rcp(new LayeredMeshNumbering<LO>(lVertexColumnShift,Ordering,layerThicknessRatio)):
            Teuchos::rcp(new LayeredMeshNumbering<LO>(vertexLayerShift,Ordering,layerThicknessRatio));


  metaData->commit();

  bulkData->modification_begin(); // Begin modifying the mesh

  stk::mesh::PartVector nodePartVec;
  stk::mesh::PartVector singlePartVec(1);
  stk::mesh::PartVector emptyPartVec;
  std::cout << "elem_map # elments: " << elem_mapT->getNodeNumElements() << std::endl;
  unsigned int ebNo = 0; //element block #???

  singlePartVec[0] = nsPartVec["bottom"];


  AbstractSTKFieldContainer::IntScalarFieldType* proc_rank_field = fieldContainer->getProcRankField();
  AbstractSTKFieldContainer::VectorFieldType* coordinates_field = fieldContainer->getCoordinatesField();
  stk::mesh::Field<double>* surfaceHeight_field = metaData->get_field<stk::mesh::Field<double> >(stk::topology::NODE_RANK, "surface_height");

  for(int i=0; i< (numLayers+1)*indexToVertexID.size(); i++)
  {
	  int ib = (Ordering == LAYER)*(i%lVertexColumnShift) + (Ordering == COLUMN)*(i/vertexLayerShift);
	  int il = (Ordering == LAYER)*(i/lVertexColumnShift) + (Ordering == COLUMN)*(i%vertexLayerShift);

    stk::mesh::Entity node;
	  if(il == 0)
		  node = bulkData->declare_entity(stk::topology::NODE_RANK, il*vertexColumnShift+vertexLayerShift * indexToVertexID[ib]+1, singlePartVec);
	  else
		  node = bulkData->declare_entity(stk::topology::NODE_RANK, il*vertexColumnShift+vertexLayerShift * indexToVertexID[ib]+1, nodePartVec);

    std::vector<int> sharing_procs;
    procsSharingVertex(ib, sharing_procs);
    for(int iproc=0; iproc<sharing_procs.size(); ++iproc)
      bulkData->add_node_sharing(node, sharing_procs[iproc]);

    double* coord = stk::mesh::field_data(*coordinates_field, node);
	  coord[0] = verticesCoords[3*ib];   coord[1] = verticesCoords[3*ib+1]; coord[2] = double(il)/numLayers;

	  double* sHeight;
	   sHeight = stk::mesh::field_data(*surfaceHeight_field, node);
	   sHeight[0] = 1.;
  }


  singlePartVec[0] = nsPartVec["dirichlet"];
  for(int i=0; i<dirichletNodesIds.size(); ++i)
  {
    stk::mesh::Entity node = bulkData->get_entity(stk::topology::NODE_RANK, dirichletNodesIds[i]+1);
    bulkData->change_entity_parts(node, singlePartVec);
  }

  for (int i=0; i<elem_mapT->getNodeNumElements(); i++) {

	 int ib = (Ordering == LAYER)*(i%lElemColumnShift) + (Ordering == COLUMN)*(i/elemLayerShift);
	 int il = (Ordering == LAYER)*(i/lElemColumnShift) + (Ordering == COLUMN)*(i%elemLayerShift);

	 int shift = il*vertexColumnShift;

	 singlePartVec[0] = partVec[ebNo];
     stk::mesh::Entity elem  = bulkData->declare_entity(stk::topology::ELEMENT_RANK, elem_mapT->getGlobalElement(i)+1, singlePartVec);

     for(int j=0; j<3; j++)
     {
    	 int lowerId = shift+vertexLayerShift * indexToVertexID[verticesOnTria[3*ib+j]]+1;
    	 stk::mesh::Entity node = bulkData->get_entity(stk::topology::NODE_RANK, lowerId);
    	 bulkData->declare_relation(elem, node, j);

    	 stk::mesh::Entity node_top = bulkData->get_entity(stk::topology::NODE_RANK, lowerId+vertexColumnShift);
    	 bulkData->declare_relation(elem, node_top, j+3);
     }

     int* p_rank = (int*)stk::mesh::field_data(*proc_rank_field, elem);
     p_rank[0] = commT->getRank();
  }


  singlePartVec[0] = ssPartVec["lateralside"];

  //first we store the lateral faces of prisms, which corresponds to edges of the basal mesh

  for (int i=0; i<indexToEdgeID.size()*numLayers; i++) {
	 int ib = (Ordering == LAYER)*(i%lEdgeColumnShift) + (Ordering == COLUMN)*(i/edgeLayerShift);
	 if(isBoundaryEdge[ib])
	 {
		 int il = (Ordering == LAYER)*(i/lEdgeColumnShift) + (Ordering == COLUMN)*(i%edgeLayerShift);
		 int basalEdgeId = indexToEdgeID[ib]*edgeLayerShift;
		 int basalElemId = indexToTriangleID[trianglesOnEdge[2*ib]]*elemLayerShift;
		 int basalVertexId[2] = {indexToVertexID[verticesOnEdge[2*ib]]*vertexLayerShift, indexToVertexID[verticesOnEdge[2*ib+1]]*vertexLayerShift};
		 stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), edgeColumnShift*il+basalEdgeId+1, singlePartVec);
		 stk::mesh::Entity elem  = bulkData->get_entity(stk::topology::ELEMENT_RANK,  basalElemId+elemColumnShift*il+1);
		 bulkData->declare_relation(elem, side,  trianglesPositionsOnEdge[2*ib] );
		 for(int j=0; j<2; j++)
		 {
			 stk::mesh::Entity nodeBottom = bulkData->get_entity(stk::topology::NODE_RANK, basalVertexId[j]+vertexColumnShift*il+1);
			 bulkData->declare_relation(side, nodeBottom, j);
			 stk::mesh::Entity nodeTop = bulkData->get_entity(stk::topology::NODE_RANK, basalVertexId[j]+vertexColumnShift*(il+1)+1);
			 bulkData->declare_relation(side, nodeTop, j+2);
		 }
	 }
  }


  singlePartVec[0] = ssPartVec["floatinglateralside"];
  for(int i=0; i<floating2dLateralEdgesIds.size(); ++i) {
    int basalEdgeId = floating2dLateralEdgesIds[i]*edgeLayerShift;;
    for(int il=0; il<numLayers; ++il) {
    int sideId = edgeColumnShift*il+ basalEdgeId+1;
    stk::mesh::Entity side = bulkData->get_entity(metaData->side_rank(), sideId);
    bulkData->change_entity_parts(side, singlePartVec);
    }
  }

  //then we store the lower and upper faces of prisms, which corresponds to triangles of the basal mesh

  edgeLayerShift = (Ordering == LAYER) ? 1 : numLayers+1;
  edgeColumnShift = elemColumnShift;

  singlePartVec[0] = ssPartVec["basalside"];

  int edgeOffset = nGlobalEdges*numLayers;
  for (int i=0; i<indexToTriangleID.size(); i++)
  {
	  stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), indexToTriangleID[i]*edgeLayerShift+edgeOffset+1, singlePartVec);
	  stk::mesh::Entity elem  = bulkData->get_entity(stk::topology::ELEMENT_RANK,  indexToTriangleID[i]*elemLayerShift+1);
	  bulkData->declare_relation(elem, side,  3);
	  for(int j=0; j<3; j++)
	  {
		 stk::mesh::Entity node = bulkData->get_entity(stk::topology::NODE_RANK, vertexLayerShift*indexToVertexID[verticesOnTria[3*i+j]]+1);
		 bulkData->declare_relation(side, node, j);
	  }
  }

  singlePartVec[0] = ssPartVec["upperside"];

  for (int i=0; i<indexToTriangleID.size(); i++)
  {
  	  stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), indexToTriangleID[i]*edgeLayerShift+numLayers*edgeColumnShift+edgeOffset+1, singlePartVec);
  	  stk::mesh::Entity elem  = bulkData->get_entity(stk::topology::ELEMENT_RANK,  indexToTriangleID[i]*elemLayerShift+(numLayers-1)*elemColumnShift+1);
  	  bulkData->declare_relation(elem, side,  4);
  	  for(int j=0; j<3; j++)
  	  {
  		 stk::mesh::Entity node = bulkData->get_entity(stk::topology::NODE_RANK, vertexLayerShift*indexToVertexID[verticesOnTria[3*i+j]]+numLayers*vertexColumnShift+1);
  		 bulkData->declare_relation(side, node, j);
  	  }
  }

  //Albany::fix_node_sharing(*bulkData);
  bulkData->modification_end();
}



void
Albany::MpasSTKMeshStruct::constructMesh(
                                               const Teuchos::RCP<const Teuchos_Comm>& commT,
                                               const Teuchos::RCP<Teuchos::ParameterList>& params,
                                               const unsigned int neq_,
                                               const Albany::AbstractFieldContainer::FieldContainerRequirements& req,
                                               const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                                               const std::vector<int>& indexToVertexID, const std::vector<int>& indexToMpasVertexID, const std::vector<double>& verticesCoords, const std::vector<bool>& isVertexBoundary, int nGlobalVertices,
                                               const std::vector<int>& verticesOnTria,
                                               const std::vector<bool>& isBoundaryEdge, const std::vector<int>& trianglesOnEdge, const std::vector<int>& trianglesPositionsOnEdge,
                                               const std::vector<int>& verticesOnEdge,
                                               const std::vector<int>& indexToEdgeID, int nGlobalEdges,
                                               const std::vector<GO>& indexToTriangleID,
                                               const std::vector<int>& dirichletNodesIds,
                                               const std::vector<int>& floating2dLateralEdgesIds,
                                               const unsigned int worksetSize,
                                               int numLayers, int ordering)
{
	this->SetupFieldData(commT, neq_, req, sis, worksetSize);

  auto LAYER  = LayeredMeshOrdering::LAYER;
  auto COLUMN = LayeredMeshOrdering::COLUMN;

  Ordering = (ordering==0) ? LAYER : COLUMN;

  int elemColumnShift = (Ordering == COLUMN) ? 3 : elem_mapT->getGlobalNumElements()/numLayers;
  int lElemColumnShift = (Ordering == COLUMN) ? 3 : 3*indexToTriangleID.size();
  int elemLayerShift = (Ordering == LAYER) ? 3 : 3*numLayers;

  int vertexColumnShift = (Ordering == COLUMN) ? 1 : nGlobalVertices;
  int lVertexColumnShift = (Ordering == COLUMN) ? 1 : indexToVertexID.size();
  int vertexLayerShift = (Ordering == LAYER) ? 1 : numLayers+1;

  int edgeColumnShift = (Ordering == COLUMN) ? 2 : 2*nGlobalEdges;
  int lEdgeColumnShift = (Ordering == COLUMN) ? 1 : indexToEdgeID.size();
  int edgeLayerShift = (Ordering == LAYER) ? 1 : numLayers;

  Teuchos::ArrayRCP<double> layerThicknessRatio(numLayers, 1.0/double(numLayers));
  this->layered_mesh_numbering = (Ordering == LAYER) ?
        Teuchos::rcp(new LayeredMeshNumbering<LO>(lVertexColumnShift,Ordering,layerThicknessRatio)):
        Teuchos::rcp(new LayeredMeshNumbering<LO>(vertexLayerShift,Ordering,layerThicknessRatio));


  metaData->commit();

  bulkData->modification_begin(); // Begin modifying the mesh

  stk::mesh::PartVector nodePartVec;
  stk::mesh::PartVector singlePartVec(1);
  stk::mesh::PartVector emptyPartVec;
  std::cout << "elem_map # elments: " << elem_mapT->getNodeNumElements() << std::endl;
  unsigned int ebNo = 0; //element block #???

  singlePartVec[0] = nsPartVec["bottom"];

  AbstractSTKFieldContainer::IntScalarFieldType* proc_rank_field = fieldContainer->getProcRankField();
  AbstractSTKFieldContainer::VectorFieldType* coordinates_field = fieldContainer->getCoordinatesField();

  for(int i=0; i< (numLayers+1)*indexToVertexID.size(); i++)
  {
	  int ib = (Ordering == LAYER)*(i%lVertexColumnShift) + (Ordering == COLUMN)*(i/vertexLayerShift);
	  int il = (Ordering == LAYER)*(i/lVertexColumnShift) + (Ordering == COLUMN)*(i%vertexLayerShift);

	  stk::mesh::Entity node;
	  if(il == 0)
		  node = bulkData->declare_entity(stk::topology::NODE_RANK, il*vertexColumnShift+vertexLayerShift * indexToVertexID[ib]+1, singlePartVec);
	  else
		  node = bulkData->declare_entity(stk::topology::NODE_RANK, il*vertexColumnShift+vertexLayerShift * indexToVertexID[ib]+1, nodePartVec);

    std::vector<int> sharing_procs;
    procsSharingVertex(ib, sharing_procs);
    for(int iproc=0; iproc<sharing_procs.size(); ++iproc)
      bulkData->add_node_sharing(node, sharing_procs[iproc]);

      double* coord = stk::mesh::field_data(*coordinates_field, node);
	  coord[0] = verticesCoords[3*ib];   coord[1] = verticesCoords[3*ib+1]; coord[2] = double(il)/numLayers;
  }

  singlePartVec[0] = nsPartVec["dirichlet"];
  for(int i=0; i<dirichletNodesIds.size(); ++i)
  {
    stk::mesh::Entity node = bulkData->get_entity(stk::topology::NODE_RANK, dirichletNodesIds[i]+1);
    bulkData->change_entity_parts(node, singlePartVec);
  }

  int tetrasLocalIdsOnPrism[3][4];

  for (int i=0; i<elem_mapT->getNodeNumElements()/3; i++) {

	 int ib = (Ordering == LAYER)*(i%(lElemColumnShift/3)) + (Ordering == COLUMN)*(i/(elemLayerShift/3));
	 int il = (Ordering == LAYER)*(i/(lElemColumnShift/3)) + (Ordering == COLUMN)*(i%(elemLayerShift/3));

	 int shift = il*vertexColumnShift;

	 singlePartVec[0] = partVec[ebNo];


     //TODO: this could be done only in the first layer and then copied into the other layers
     int prismMpasIds[3], prismGlobalIds[6];
     for (int j = 0; j < 3; j++)
	 {
    	 int mpasLowerId = vertexLayerShift * indexToMpasVertexID[verticesOnTria[3*ib+j]];
    	 int lowerId = shift+vertexLayerShift * indexToVertexID[verticesOnTria[3*ib+j]];
    	 prismMpasIds[j] = mpasLowerId;
		 prismGlobalIds[j] = lowerId;
		 prismGlobalIds[j + 3] = lowerId+vertexColumnShift;
	 }

     tetrasFromPrismStructured (prismMpasIds, prismGlobalIds, tetrasLocalIdsOnPrism);


     for(int iTetra = 0; iTetra<3; iTetra++)
     {
    	 stk::mesh::Entity elem  = bulkData->declare_entity(stk::topology::ELEMENT_RANK, elem_mapT->getGlobalElement(3*i+iTetra)+1, singlePartVec);
		 for(int j=0; j<4; j++)
		 {
			 stk::mesh::Entity node = bulkData->get_entity(stk::topology::NODE_RANK, tetrasLocalIdsOnPrism[iTetra][j]+1);
			 bulkData->declare_relation(elem, node, j);
		 }
		 int* p_rank = (int*)stk::mesh::field_data(*proc_rank_field, elem);
		 p_rank[0] = commT->getRank();
     }


  }


  singlePartVec[0] = ssPartVec["lateralside"];

  //first we store the lateral faces of prisms, which corresponds to edges of the basal mesh
  int tetraSidePoints[4][3] = {{0, 1, 3}, {1, 2, 3}, {0, 3, 2}, {0, 2, 1}};
  std::vector<int> tetraPos(2), facePos(2);

  std::vector<std::vector<std::vector<int> > > prismStruct(3, std::vector<std::vector<int> >(4, std::vector<int>(3)));
  for (int i=0; i<indexToEdgeID.size()*numLayers; i++) {
	 int ib = (Ordering == LAYER)*(i%lEdgeColumnShift) + (Ordering == COLUMN)*(i/edgeLayerShift);
	 if(isBoundaryEdge[ib])
	 {
		 int il = (Ordering == LAYER)*(i/lEdgeColumnShift) + (Ordering == COLUMN)*(i%edgeLayerShift);
		 int lBasalElemId = trianglesOnEdge[2*ib];
		 int basalElemId = indexToTriangleID[lBasalElemId];

		 //TODO: this could be done only in the first layer and then copied into the other layers
		 int prismMpasIds[3], prismGlobalIds[6];
		 int shift = il*vertexColumnShift;
		 for (int j = 0; j < 3; j++)
		 {
			 int mpasLowerId = vertexLayerShift * indexToMpasVertexID[verticesOnTria[3*lBasalElemId+j]];
			 int lowerId = shift+vertexLayerShift * indexToVertexID[verticesOnTria[3*lBasalElemId+j]];
			 prismMpasIds[j] = mpasLowerId;
			 prismGlobalIds[j] = lowerId;
			 prismGlobalIds[j + 3] = lowerId+vertexColumnShift;
		 }

		  tetrasFromPrismStructured (prismMpasIds, prismGlobalIds, tetrasLocalIdsOnPrism);


		for(int iTetra = 0; iTetra<3; iTetra++)
		  {
			 std::vector<std::vector<int> >& tetraStruct =prismStruct[iTetra];
			 stk::mesh::EntityId tetraPoints[4];
			 for(int j=0; j<4; j++)
			 {
                           tetraPoints[j] = tetrasLocalIdsOnPrism[iTetra][j]+1;
				// std::cout<< tetraPoints[j] << ", ";
			 }
			 for(int iFace=0; iFace<4; iFace++)
			 {
				 std::vector<int>&  face = tetraStruct[iFace];
				 for(int j=0; j<3; j++)
				 	 face[j] = tetraPoints[tetraSidePoints[iFace][j]];
			 }
		  }



		 int basalVertexId[2] = {indexToVertexID[verticesOnEdge[2*ib]]*vertexLayerShift, indexToVertexID[verticesOnEdge[2*ib+1]]*vertexLayerShift};
		 std::vector<int> bdPrismFaceIds(4);

		 bdPrismFaceIds[0] = indexToVertexID[verticesOnEdge[2*ib]]*vertexLayerShift+vertexColumnShift*il+1;
		 bdPrismFaceIds[1] = indexToVertexID[verticesOnEdge[2*ib+1]]*vertexLayerShift+vertexColumnShift*il+1;
		 bdPrismFaceIds[2] = bdPrismFaceIds[0]+vertexColumnShift;
		 bdPrismFaceIds[3] = bdPrismFaceIds[1]+vertexColumnShift;

		 //std::cout<< "bdPrismFaceIds: (" << bdPrismFaceIds[0] << ", " << bdPrismFaceIds[1] << ", " << bdPrismFaceIds[2] << ", " << bdPrismFaceIds[3] << ")"<<std::endl;



		 setBdFacesOnPrism (prismStruct, bdPrismFaceIds, tetraPos, facePos);

		 int basalEdgeId = indexToEdgeID[ib]*2*edgeLayerShift;
		 for(int k=0; k< tetraPos.size(); k++)
		 {
			 int iTetra = tetraPos[k];
			 int iFace = facePos[k];
			 stk::mesh::Entity elem = bulkData->get_entity(stk::topology::ELEMENT_RANK, il*elemColumnShift+elemLayerShift * basalElemId +iTetra+1);
			 std::vector<int>& faceIds = prismStruct[iTetra][iFace];
			 stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), edgeColumnShift*il+basalEdgeId+k+1, singlePartVec);
			 bulkData->declare_relation(elem, side,  iFace );
			 for(int j=0; j<3; j++)
			 {
				 stk::mesh::Entity node = bulkData->get_entity(stk::topology::NODE_RANK, faceIds[j]);
				 bulkData->declare_relation(side, node, j);
			 }
		 }
	 }
  }

  singlePartVec[0] = ssPartVec["floatinglateralside"];
  for(int i=0; i<floating2dLateralEdgesIds.size(); ++i) {
    int basalEdgeId = floating2dLateralEdgesIds[i]*2*edgeLayerShift;
    for(int il=0; il<numLayers; ++il)
      for(int k=0; k< 2; k++){
        int sideId = edgeColumnShift*il+basalEdgeId+k+1;
        stk::mesh::Entity side = bulkData->get_entity(metaData->side_rank(), sideId);
        bulkData->change_entity_parts(side, singlePartVec);
      }
  }

  //then we store the lower and upper faces of prisms, which corresponds to triangles of the basal mesh

  edgeLayerShift = (Ordering == LAYER) ? 1 : numLayers+1;
  edgeColumnShift = 2*(elemColumnShift/3);

  singlePartVec[0] = ssPartVec["basalside"];

  int edgeOffset = 2*nGlobalEdges*numLayers;
  for (int i=0; i<indexToTriangleID.size(); i++)
  {
	  stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), indexToTriangleID[i]*edgeLayerShift+edgeOffset+1, singlePartVec);
	  stk::mesh::Entity elem  = bulkData->get_entity(stk::topology::ELEMENT_RANK,  indexToTriangleID[i]*elemLayerShift+1);
	  bulkData->declare_relation(elem, side,  3);
	  for(int j=0; j<3; j++)
	  {
		 stk::mesh::Entity node = bulkData->get_entity(stk::topology::NODE_RANK, vertexLayerShift*indexToVertexID[verticesOnTria[3*i+j]]+1);
		 bulkData->declare_relation(side, node, j);
	  }
  }

  singlePartVec[0] = ssPartVec["upperside"];

  for (int i=0; i<indexToTriangleID.size(); i++)
  {
	stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), indexToTriangleID[i]*edgeLayerShift+numLayers*edgeColumnShift+edgeOffset+1, singlePartVec);
	stk::mesh::Entity elem  = bulkData->get_entity(stk::topology::ELEMENT_RANK,  indexToTriangleID[i]*elemLayerShift+(numLayers-1)*elemColumnShift+1+2);
	bulkData->declare_relation(elem, side,  1);
	for(int j=0; j<3; j++)
	{
	  stk::mesh::Entity node = bulkData->get_entity(stk::topology::NODE_RANK, vertexLayerShift*indexToVertexID[verticesOnTria[3*i+j]]+numLayers*vertexColumnShift+1);
	  bulkData->declare_relation(side, node, j);
	}
  }

  //Albany::fix_node_sharing(*bulkData);
  bulkData->modification_end();
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::MpasSTKMeshStruct::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getValidGenericSTKParameters("Valid ASCII_DiscParams");

  return validPL;
}
