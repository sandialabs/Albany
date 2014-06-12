//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>

#include "Albany_MpasSTKMeshStruct.hpp"
#include "Teuchos_VerboseObject.hpp"

#include <Shards_BasicTopologies.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/FieldData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_io/IossBridge.hpp>
#include <Ioss_SubSystem.h>

//#include <stk_mesh/fem/FEMHelpers.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "Albany_Utils.hpp"

Albany::MpasSTKMeshStruct::MpasSTKMeshStruct(const Teuchos::RCP<Teuchos::ParameterList>& params,
                                             const Teuchos::RCP<const Epetra_Comm>& comm,
                                             const std::vector<int>& indexToTriangleID, const std::vector<int>& verticesOnTria, int nGlobalTriangles) :
  GenericSTKMeshStruct(params,Teuchos::null, 2),
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  periodic(false),
  NumEles(indexToTriangleID.size()),
  hasRestartSol(false),
  restartTime(0.)
{
  elem_map = Teuchos::rcp(new Epetra_Map(nGlobalTriangles, indexToTriangleID.size(), &indexToTriangleID[0], 0, *comm)); // Distribute the elems equally
  
  params->validateParameters(*getValidDiscretizationParameters(),0);


  std::string ebn="Element Block 0";
  partVec[0] = & metaData->declare_part(ebn, metaData->element_rank() );
  ebNameToIndex[ebn] = 0;

#ifdef ALBANY_SEACAS
  stk_classic::io::put_io_part_attribute(*partVec[0]);
#endif


  std::vector<std::string> nsNames;
  std::string nsn="Lateral";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
    stk_classic::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn="Internal";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
    stk_classic::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif


  std::vector<std::string> ssNames;
  std::string ssn="LateralSide";
  ssNames.push_back(ssn);
    ssPartVec[ssn] = & metaData->declare_part(ssn, metaData->side_rank() );
#ifdef ALBANY_SEACAS
    stk_classic::io::put_io_part_attribute(*ssPartVec[ssn]);
#endif

  stk_classic::mesh::fem::set_cell_topology<shards::Triangle<3> >(*partVec[0]);
  stk_classic::mesh::fem::set_cell_topology<shards::Line<2> >(*ssPartVec[ssn]);

  numDim = 2;
  int cub = params->get("Cubature Degree",3);
  int worksetSizeMax = params->get("Workset Size",50);
  int worksetSize = this->computeWorksetSize(worksetSizeMax, elem_map->NumMyElements());

  const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();

  this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub,
                             nsNames, ssNames, worksetSize, partVec[0]->name(),
                             ebNameToIndex, this->interleavedOrdering));


}

//Wedge
Albany::MpasSTKMeshStruct::MpasSTKMeshStruct(const Teuchos::RCP<Teuchos::ParameterList>& params,
                                             const Teuchos::RCP<const Epetra_Comm>& comm,
                                             const std::vector<int>& indexToTriangleID, const std::vector<int>& verticesOnTria, int nGlobalTriangles, int numLayers, int Ordering) :
  GenericSTKMeshStruct(params,Teuchos::null,3),
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  periodic(false),
  NumEles(indexToTriangleID.size()),
  hasRestartSol(false),
  restartTime(0.)
{
  std::vector<int> indexToPrismID(indexToTriangleID.size()*numLayers);

  //Int ElemColumnShift = (ordering == ColumnWise) ? 1 : indexToTriangleID.size();
  int elemColumnShift = (Ordering == 1) ? 1 : nGlobalTriangles;
  int lElemColumnShift = (Ordering == 1) ? 1 : indexToTriangleID.size();
  int elemLayerShift = (Ordering == 0) ? 1 : numLayers;

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

  elem_map = Teuchos::rcp(new Epetra_Map(nGlobalTriangles*numLayers, indexToPrismID.size(), &indexToPrismID[0], 0, *comm)); // Distribute the elems equally

  params->validateParameters(*getValidDiscretizationParameters(),0);


  std::string ebn="Element Block 0";
  partVec[0] = & metaData->declare_part(ebn, metaData->element_rank() );
  ebNameToIndex[ebn] = 0;

#ifdef ALBANY_SEACAS
  stk_classic::io::put_io_part_attribute(*partVec[0]);
#endif


  std::vector<std::string> nsNames;
  std::string nsn="Lateral";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
    stk_classic::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn="Internal";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
    stk_classic::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn="Bottom";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
	stk_classic::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif


  std::vector<std::string> ssNames;
  std::string ssnLat="lateralside";
  std::string ssnBottom="basalside";
  std::string ssnTop="upperside";

  ssNames.push_back(ssnLat);
  ssNames.push_back(ssnBottom);
  ssNames.push_back(ssnTop);
  ssPartVec[ssnLat] = & metaData->declare_part(ssnLat, metaData->side_rank() );
  ssPartVec[ssnBottom] = & metaData->declare_part(ssnBottom, metaData->side_rank() );
  ssPartVec[ssnTop] = & metaData->declare_part(ssnTop, metaData->side_rank() );
#ifdef ALBANY_SEACAS
    stk_classic::io::put_io_part_attribute(*ssPartVec[ssnLat]);
    stk_classic::io::put_io_part_attribute(*ssPartVec[ssnBottom]);
    stk_classic::io::put_io_part_attribute(*ssPartVec[ssnTop]);
#endif

  stk_classic::mesh::fem::set_cell_topology<shards::Wedge<6> >(*partVec[0]);
  stk_classic::mesh::fem::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnBottom]);
  stk_classic::mesh::fem::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnTop]);
  stk_classic::mesh::fem::set_cell_topology<shards::Quadrilateral<4> >(*ssPartVec[ssnLat]);

  numDim = 3;
  int cub = params->get("Cubature Degree",3);
  int worksetSizeMax = params->get("Workset Size",50);
  int worksetSize = this->computeWorksetSize(worksetSizeMax, elem_map->NumMyElements());

  const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();

  this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub,
                             nsNames, ssNames, worksetSize, partVec[0]->name(),
                             ebNameToIndex, this->interleavedOrdering));


}

//Tetra
Albany::MpasSTKMeshStruct::MpasSTKMeshStruct(const Teuchos::RCP<Teuchos::ParameterList>& params,
                                             const Teuchos::RCP<const Epetra_Comm>& comm,
                                             const std::vector<int>& indexToTriangleID, int nGlobalTriangles, int numLayers, int Ordering) :
  GenericSTKMeshStruct(params,Teuchos::null,3),
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  periodic(false),
  NumEles(indexToTriangleID.size()),
  hasRestartSol(false),
  restartTime(0.)
{
  std::vector<int> indexToTetraID(3*indexToTriangleID.size()*numLayers);

  //Int ElemColumnShift = (ordering == ColumnWise) ? 1 : indexToTriangleID.size();
  int elemColumnShift = (Ordering == 1) ? 3 : 3*nGlobalTriangles;
  int lElemColumnShift = (Ordering == 1) ? 3 : 3*indexToTriangleID.size();
  int elemLayerShift = (Ordering == 0) ? 3 : 3*numLayers;

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

  elem_map = Teuchos::rcp(new Epetra_Map(3*nGlobalTriangles*numLayers, indexToTetraID.size(), &indexToTetraID[0], 0, *comm)); // Distribute the elems equally

  params->validateParameters(*getValidDiscretizationParameters(),0);


  std::string ebn="Element Block 0";
  partVec[0] = & metaData->declare_part(ebn, metaData->element_rank() );
  ebNameToIndex[ebn] = 0;

#ifdef ALBANY_SEACAS
  stk_classic::io::put_io_part_attribute(*partVec[0]);
#endif


  std::vector<std::string> nsNames;
  std::string nsn="Lateral";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
    stk_classic::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn="Internal";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
    stk_classic::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn="Bottom";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
	stk_classic::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif


  std::vector<std::string> ssNames;
  std::string ssnLat="lateralside";
  std::string ssnBottom="basalside";
  std::string ssnTop="upperside";

  ssNames.push_back(ssnLat);
  ssNames.push_back(ssnBottom);
  ssNames.push_back(ssnTop);
  ssPartVec[ssnLat] = & metaData->declare_part(ssnLat, metaData->side_rank() );
  ssPartVec[ssnBottom] = & metaData->declare_part(ssnBottom, metaData->side_rank() );
  ssPartVec[ssnTop] = & metaData->declare_part(ssnTop, metaData->side_rank() );
#ifdef ALBANY_SEACAS
    stk_classic::io::put_io_part_attribute(*ssPartVec[ssnLat]);
    stk_classic::io::put_io_part_attribute(*ssPartVec[ssnBottom]);
    stk_classic::io::put_io_part_attribute(*ssPartVec[ssnTop]);
#endif

  stk_classic::mesh::fem::set_cell_topology<shards::Tetrahedron<4> >(*partVec[0]);
  stk_classic::mesh::fem::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnBottom]);
  stk_classic::mesh::fem::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnTop]);
  stk_classic::mesh::fem::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnLat]);

  numDim = 3;
  int cub = params->get("Cubature Degree",3);
  int worksetSizeMax = params->get("Workset Size",50);
  int worksetSize = this->computeWorksetSize(worksetSizeMax, elem_map->NumMyElements());

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
                                               const Teuchos::RCP<const Epetra_Comm>& comm,
                                               const Teuchos::RCP<Teuchos::ParameterList>& params,
                                               const unsigned int neq_,
                                               const Albany::AbstractFieldContainer::FieldContainerRequirements& req,
                                               const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                                               const std::vector<int>& indexToVertexID, const std::vector<double>& verticesCoords, const std::vector<bool>& isVertexBoundary, int nGlobalVertices,
                                               const std::vector<int>& verticesOnTria,
                                               const std::vector<bool>& isBoundaryEdge, const std::vector<int>& trianglesOnEdge, const std::vector<int>& trianglesPositionsOnEdge,
                                               const std::vector<int>& verticesOnEdge,
                                               const std::vector<int>& indexToEdgeID, int nGlobalEdges,
                                               const std::vector<int>& indexToTriangleID,
                                               const unsigned int worksetSize,
                                               int numLayers, int Ordering)
{
	this->SetupFieldData(comm, neq_, req, sis, worksetSize);

    int elemColumnShift = (Ordering == 1) ? 1 : elem_map->NumGlobalElements()/numLayers;
    int lElemColumnShift = (Ordering == 1) ? 1 : indexToTriangleID.size();
    int elemLayerShift = (Ordering == 0) ? 1 : numLayers;

    int vertexColumnShift = (Ordering == 1) ? 1 : nGlobalVertices;
    int lVertexColumnShift = (Ordering == 1) ? 1 : indexToVertexID.size();
    int vertexLayerShift = (Ordering == 0) ? 1 : numLayers+1;

    int edgeColumnShift = (Ordering == 1) ? 1 : nGlobalEdges;
    int lEdgeColumnShift = (Ordering == 1) ? 1 : indexToEdgeID.size();
    int edgeLayerShift = (Ordering == 0) ? 1 : numLayers;


  metaData->commit();

  bulkData->modification_begin(); // Begin modifying the mesh

  stk_classic::mesh::PartVector nodePartVec;
  stk_classic::mesh::PartVector singlePartVec(1);
  stk_classic::mesh::PartVector emptyPartVec;
  std::cout << "elem_map # elments: " << elem_map->NumMyElements() << std::endl;
  unsigned int ebNo = 0; //element block #???

  singlePartVec[0] = nsPartVec["Bottom"];


  AbstractSTKFieldContainer::IntScalarFieldType* proc_rank_field = fieldContainer->getProcRankField();
  AbstractSTKFieldContainer::VectorFieldType* coordinates_field = fieldContainer->getCoordinatesField();
  AbstractSTKFieldContainer::ScalarFieldType* surfaceHeight_field = fieldContainer->getSurfaceHeightField();

  for(int i=0; i< (numLayers+1)*indexToVertexID.size(); i++)
  {
	  int ib = (Ordering == 0)*(i%lVertexColumnShift) + (Ordering == 1)*(i/vertexLayerShift);
	  int il = (Ordering == 0)*(i/lVertexColumnShift) + (Ordering == 1)*(i%vertexLayerShift);

	  stk_classic::mesh::Entity* node;
	  if(il == 0)
		  node = &bulkData->declare_entity(metaData->node_rank(), il*vertexColumnShift+vertexLayerShift * indexToVertexID[ib]+1, singlePartVec);
	  else
		  node = &bulkData->declare_entity(metaData->node_rank(), il*vertexColumnShift+vertexLayerShift * indexToVertexID[ib]+1, nodePartVec);
	  int numBdEdges(0);
	  for (int i=0; i<indexToEdgeID.size(); i++)
		  numBdEdges += isBoundaryEdge[i];


      double* coord = stk_classic::mesh::field_data(*coordinates_field, *node);
	  coord[0] = verticesCoords[3*ib];   coord[1] = verticesCoords[3*ib+1]; coord[2] = double(il)/numLayers;

	  double* sHeight;
	   sHeight = stk_classic::mesh::field_data(*surfaceHeight_field, *node);
	   sHeight[0] = 1.;
  }

  for (int i=0; i<elem_map->NumMyElements(); i++) {

	 int ib = (Ordering == 0)*(i%lElemColumnShift) + (Ordering == 1)*(i/elemLayerShift);
	 int il = (Ordering == 0)*(i/lElemColumnShift) + (Ordering == 1)*(i%elemLayerShift);

	 int shift = il*vertexColumnShift;

	 singlePartVec[0] = partVec[ebNo];
     stk_classic::mesh::Entity& elem  = bulkData->declare_entity(metaData->element_rank(), elem_map->GID(i)+1, singlePartVec);

     for(int j=0; j<3; j++)
     {
    	 int lowerId = shift+vertexLayerShift * indexToVertexID[verticesOnTria[3*ib+j]]+1;
    	 stk_classic::mesh::Entity& node = *bulkData->get_entity(metaData->node_rank(), lowerId);
    	 bulkData->declare_relation(elem, node, j);

    	 stk_classic::mesh::Entity& node_top = *bulkData->get_entity(metaData->node_rank(), lowerId+vertexColumnShift);
    	 bulkData->declare_relation(elem, node_top, j+3);
     }

     int* p_rank = (int*)stk_classic::mesh::field_data(*proc_rank_field, elem);
     p_rank[0] = comm->MyPID();
  }


  singlePartVec[0] = ssPartVec["lateralside"];

  //first we store the lateral faces of prisms, which corresponds to edges of the basal mesh

  for (int i=0; i<indexToEdgeID.size()*numLayers; i++) {
	 int ib = (Ordering == 0)*(i%lEdgeColumnShift) + (Ordering == 1)*(i/edgeLayerShift);
	 if(isBoundaryEdge[ib])
	 {
		 int il = (Ordering == 0)*(i/lEdgeColumnShift) + (Ordering == 1)*(i%edgeLayerShift);
		 int basalEdgeId = indexToEdgeID[ib]*edgeLayerShift;
		 int basalElemId = indexToTriangleID[trianglesOnEdge[2*ib]]*elemLayerShift;
		 int basalVertexId[2] = {indexToVertexID[verticesOnEdge[2*ib]]*vertexLayerShift, indexToVertexID[verticesOnEdge[2*ib+1]]*vertexLayerShift};
		 stk_classic::mesh::Entity& side = bulkData->declare_entity(metaData->side_rank(), edgeColumnShift*il+basalEdgeId+1, singlePartVec);
		 stk_classic::mesh::Entity& elem  = *bulkData->get_entity(metaData->element_rank(),  basalElemId+elemColumnShift*il+1);
		 bulkData->declare_relation(elem, side,  trianglesPositionsOnEdge[2*ib] );
		 for(int j=0; j<2; j++)
		 {
			 stk_classic::mesh::Entity& nodeBottom = *bulkData->get_entity(metaData->node_rank(), basalVertexId[j]+vertexColumnShift*il+1);
			 bulkData->declare_relation(side, nodeBottom, j);
			 stk_classic::mesh::Entity& nodeTop = *bulkData->get_entity(metaData->node_rank(), basalVertexId[j]+vertexColumnShift*(il+1)+1);
			 bulkData->declare_relation(side, nodeTop, j+2);
		 }
	 }
  }

  //then we store the lower and upper faces of prisms, which corresponds to triangles of the basal mesh

  edgeLayerShift = (Ordering == 0) ? 1 : numLayers+1;
  edgeColumnShift = elemColumnShift;

  singlePartVec[0] = ssPartVec["basalside"];

  int edgeOffset = nGlobalEdges*numLayers;
  for (int i=0; i<indexToTriangleID.size(); i++)
  {
	  stk_classic::mesh::Entity& side = bulkData->declare_entity(metaData->side_rank(), indexToTriangleID[i]*edgeLayerShift+edgeOffset+1, singlePartVec);
	  stk_classic::mesh::Entity& elem  = *bulkData->get_entity(metaData->element_rank(),  indexToTriangleID[i]*elemLayerShift+1);
	  bulkData->declare_relation(elem, side,  3);
	  for(int j=0; j<3; j++)
	  {
		 stk_classic::mesh::Entity& node = *bulkData->get_entity(metaData->node_rank(), vertexLayerShift*indexToVertexID[verticesOnTria[3*i+j]]+1);
		 bulkData->declare_relation(side, node, j);
	  }
  }

  singlePartVec[0] = ssPartVec["upperside"];

  for (int i=0; i<indexToTriangleID.size(); i++)
  {
  	  stk_classic::mesh::Entity& side = bulkData->declare_entity(metaData->side_rank(), indexToTriangleID[i]*edgeLayerShift+numLayers*edgeColumnShift+edgeOffset+1, singlePartVec);
  	  stk_classic::mesh::Entity& elem  = *bulkData->get_entity(metaData->element_rank(),  indexToTriangleID[i]*elemLayerShift+(numLayers-1)*elemColumnShift+1);
  	  bulkData->declare_relation(elem, side,  4);
  	  for(int j=0; j<3; j++)
  	  {
  		 stk_classic::mesh::Entity& node = *bulkData->get_entity(metaData->node_rank(), vertexLayerShift*indexToVertexID[verticesOnTria[3*i+j]]+numLayers*vertexColumnShift+1);
  		 bulkData->declare_relation(side, node, j);
  	  }
  }

  bulkData->modification_end();
}



void
Albany::MpasSTKMeshStruct::constructMesh(
                                               const Teuchos::RCP<const Epetra_Comm>& comm,
                                               const Teuchos::RCP<Teuchos::ParameterList>& params,
                                               const unsigned int neq_,
                                               const Albany::AbstractFieldContainer::FieldContainerRequirements& req,
                                               const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                                               const std::vector<int>& indexToVertexID, const std::vector<int>& indexToMpasVertexID, const std::vector<double>& verticesCoords, const std::vector<bool>& isVertexBoundary, int nGlobalVertices,
                                               const std::vector<int>& verticesOnTria,
                                               const std::vector<bool>& isBoundaryEdge, const std::vector<int>& trianglesOnEdge, const std::vector<int>& trianglesPositionsOnEdge,
                                               const std::vector<int>& verticesOnEdge,
                                               const std::vector<int>& indexToEdgeID, int nGlobalEdges,
                                               const std::vector<int>& indexToTriangleID,
                                               const unsigned int worksetSize,
                                               int numLayers, int Ordering)
{
	this->SetupFieldData(comm, neq_, req, sis, worksetSize);

    int elemColumnShift = (Ordering == 1) ? 3 : elem_map->NumGlobalElements()/numLayers;
    int lElemColumnShift = (Ordering == 1) ? 3 : 3*indexToTriangleID.size();
    int elemLayerShift = (Ordering == 0) ? 3 : 3*numLayers;

    int vertexColumnShift = (Ordering == 1) ? 1 : nGlobalVertices;
    int lVertexColumnShift = (Ordering == 1) ? 1 : indexToVertexID.size();
    int vertexLayerShift = (Ordering == 0) ? 1 : numLayers+1;

    int edgeColumnShift = (Ordering == 1) ? 2 : 2*nGlobalEdges;
    int lEdgeColumnShift = (Ordering == 1) ? 1 : indexToEdgeID.size();
    int edgeLayerShift = (Ordering == 0) ? 1 : numLayers;


  metaData->commit();

  bulkData->modification_begin(); // Begin modifying the mesh

  stk_classic::mesh::PartVector nodePartVec;
  stk_classic::mesh::PartVector singlePartVec(1);
  stk_classic::mesh::PartVector emptyPartVec;
  std::cout << "elem_map # elments: " << elem_map->NumMyElements() << std::endl;
  unsigned int ebNo = 0; //element block #???

  singlePartVec[0] = nsPartVec["Bottom"];

  AbstractSTKFieldContainer::IntScalarFieldType* proc_rank_field = fieldContainer->getProcRankField();
  AbstractSTKFieldContainer::VectorFieldType* coordinates_field = fieldContainer->getCoordinatesField();
  AbstractSTKFieldContainer::ScalarFieldType* surfaceHeight_field = fieldContainer->getSurfaceHeightField();


  for(int i=0; i< (numLayers+1)*indexToVertexID.size(); i++)
  {
	  int ib = (Ordering == 0)*(i%lVertexColumnShift) + (Ordering == 1)*(i/vertexLayerShift);
	  int il = (Ordering == 0)*(i/lVertexColumnShift) + (Ordering == 1)*(i%vertexLayerShift);

	  stk_classic::mesh::Entity* node;
	  if(il == 0)
		  node = &bulkData->declare_entity(metaData->node_rank(), il*vertexColumnShift+vertexLayerShift * indexToVertexID[ib]+1, singlePartVec);
	  else
		  node = &bulkData->declare_entity(metaData->node_rank(), il*vertexColumnShift+vertexLayerShift * indexToVertexID[ib]+1, nodePartVec);

      double* coord = stk_classic::mesh::field_data(*coordinates_field, *node);
	  coord[0] = verticesCoords[3*ib];   coord[1] = verticesCoords[3*ib+1]; coord[2] = double(il)/numLayers;

	  double* sHeight;
	   sHeight = stk_classic::mesh::field_data(*surfaceHeight_field, *node);
	   sHeight[0] = 1.;
  }

  int tetrasLocalIdsOnPrism[3][4];

  for (int i=0; i<elem_map->NumMyElements()/3; i++) {

	 int ib = (Ordering == 0)*(i%(lElemColumnShift/3)) + (Ordering == 1)*(i/(elemLayerShift/3));
	 int il = (Ordering == 0)*(i/(lElemColumnShift/3)) + (Ordering == 1)*(i%(elemLayerShift/3));

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
    	 stk_classic::mesh::Entity& elem  = bulkData->declare_entity(metaData->element_rank(), elem_map->GID(3*i+iTetra)+1, singlePartVec);
		 for(int j=0; j<4; j++)
		 {
			 stk_classic::mesh::Entity& node = *bulkData->get_entity(metaData->node_rank(), tetrasLocalIdsOnPrism[iTetra][j]+1);
			 bulkData->declare_relation(elem, node, j);
		 }
		 int* p_rank = (int*)stk_classic::mesh::field_data(*proc_rank_field, elem);
		 p_rank[0] = comm->MyPID();
     }


  }


  singlePartVec[0] = ssPartVec["lateralside"];

  //first we store the lateral faces of prisms, which corresponds to edges of the basal mesh
  int tetraSidePoints[4][3] = {{0, 1, 3}, {1, 2, 3}, {0, 3, 2}, {0, 2, 1}};
  std::vector<int> tetraPos(2), facePos(2);

  std::vector<std::vector<std::vector<int> > > prismStruct(3, std::vector<std::vector<int> >(4, std::vector<int>(3)));
  for (int i=0; i<indexToEdgeID.size()*numLayers; i++) {
	 int ib = (Ordering == 0)*(i%lEdgeColumnShift) + (Ordering == 1)*(i/edgeLayerShift);
	 if(isBoundaryEdge[ib])
	 {
		 int il = (Ordering == 0)*(i/lEdgeColumnShift) + (Ordering == 1)*(i%edgeLayerShift);
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
			 stk_classic::mesh::EntityId tetraPoints[4];
			 for(int j=0; j<4; j++)
			 {
				 tetraPoints[j] = bulkData->get_entity(metaData->node_rank(), tetrasLocalIdsOnPrism[iTetra][j]+1)->identifier();
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
			 stk_classic::mesh::Entity& elem = *bulkData->get_entity(metaData->element_rank(), il*elemColumnShift+elemLayerShift * basalElemId +iTetra+1);
			 std::vector<int>& faceIds = prismStruct[iTetra][iFace];
			 stk_classic::mesh::Entity& side = bulkData->declare_entity(metaData->side_rank(), edgeColumnShift*il+basalEdgeId+k+1, singlePartVec);
			 bulkData->declare_relation(elem, side,  iFace );
			 for(int j=0; j<3; j++)
			 {
				 stk_classic::mesh::Entity& node = *bulkData->get_entity(metaData->node_rank(), faceIds[j]);
				 bulkData->declare_relation(side, node, j);
			 }
		 }
	 }
  }

  //then we store the lower and upper faces of prisms, which corresponds to triangles of the basal mesh

  edgeLayerShift = (Ordering == 0) ? 1 : numLayers+1;
  edgeColumnShift = 2*(elemColumnShift/3);

  singlePartVec[0] = ssPartVec["basalside"];

  int edgeOffset = 2*nGlobalEdges*numLayers;
  for (int i=0; i<indexToTriangleID.size(); i++)
  {
	  stk_classic::mesh::Entity& side = bulkData->declare_entity(metaData->side_rank(), indexToTriangleID[i]*edgeLayerShift+edgeOffset+1, singlePartVec);
	  stk_classic::mesh::Entity& elem  = *bulkData->get_entity(metaData->element_rank(),  indexToTriangleID[i]*elemLayerShift+1);
	  bulkData->declare_relation(elem, side,  3);
	  for(int j=0; j<3; j++)
	  {
		 stk_classic::mesh::Entity& node = *bulkData->get_entity(metaData->node_rank(), vertexLayerShift*indexToVertexID[verticesOnTria[3*i+j]]+1);
		 bulkData->declare_relation(side, node, j);
	  }
  }

  singlePartVec[0] = ssPartVec["upperside"];

  for (int i=0; i<indexToTriangleID.size(); i++)
  {
	stk_classic::mesh::Entity& side = bulkData->declare_entity(metaData->side_rank(), indexToTriangleID[i]*edgeLayerShift+numLayers*edgeColumnShift+edgeOffset+1, singlePartVec);
	stk_classic::mesh::Entity& elem  = *bulkData->get_entity(metaData->element_rank(),  indexToTriangleID[i]*elemLayerShift+(numLayers-1)*elemColumnShift+1+2);
	bulkData->declare_relation(elem, side,  1);
	for(int j=0; j<3; j++)
	{
	  stk_classic::mesh::Entity& node = *bulkData->get_entity(metaData->node_rank(), vertexLayerShift*indexToVertexID[verticesOnTria[3*i+j]]+numLayers*vertexColumnShift+1);
	  bulkData->declare_relation(side, node, j);
	}
  }

  bulkData->modification_end();
}

void
Albany::MpasSTKMeshStruct::constructMesh(
                                               const Teuchos::RCP<const Epetra_Comm>& comm,
                                               const Teuchos::RCP<Teuchos::ParameterList>& params,
                                               const unsigned int neq_,
                                               const Albany::AbstractFieldContainer::FieldContainerRequirements& req,
                                               const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                                               const std::vector<int>& indexToVertexID, const std::vector<double>& verticesCoords, const std::vector<bool>& isVertexBoundary, int nGlobalVertices,
                                               const std::vector<int>& verticesOnTria,
                                               const std::vector<bool>& isBoundaryEdge, const std::vector<int>& trianglesOnEdge, const std::vector<int>& trianglesPositionsOnEdge,
                                               const std::vector<int>& verticesOnEdge,
                                               const std::vector<int>& indexToEdgeID, int nGlobalEdges,
                                               const unsigned int worksetSize)
{
  this->SetupFieldData(comm, neq_, req, sis, worksetSize);

  metaData->commit();

  bulkData->modification_begin(); // Begin modifying the mesh

  stk_classic::mesh::PartVector nodePartVec;
  stk_classic::mesh::PartVector singlePartVec(1);
  stk_classic::mesh::PartVector emptyPartVec;
  std::cout << "elem_map # elments: " << elem_map->NumMyElements() << std::endl;
  unsigned int ebNo = 0; //element block #??? 
  int sideID = 0;

  AbstractSTKFieldContainer::IntScalarFieldType* proc_rank_field = fieldContainer->getProcRankField();
  AbstractSTKFieldContainer::VectorFieldType* coordinates_field = fieldContainer->getCoordinatesField();
  AbstractSTKFieldContainer::ScalarFieldType* surfaceHeight_field = fieldContainer->getSurfaceHeightField();

  for (int i=0; i<indexToVertexID.size(); i++)
  {
	  stk_classic::mesh::Entity& node = bulkData->declare_entity(metaData->node_rank(), indexToVertexID[i]+1, nodePartVec);

	  double* coord;
	  coord = stk_classic::mesh::field_data(*coordinates_field, node);
	  coord[0] = verticesCoords[3*i];   coord[1] = verticesCoords[3*i+1]; coord[2] = verticesCoords[3*i+2];
  }

  for (int i=0; i<elem_map->NumMyElements(); i++)
  {

     singlePartVec[0] = partVec[ebNo];
     stk_classic::mesh::Entity& elem  = bulkData->declare_entity(metaData->element_rank(), elem_map->GID(i)+1, singlePartVec);

     for(int j=0; j<3; j++)
     {
    	 stk_classic::mesh::Entity& node = *bulkData->get_entity(metaData->node_rank(), indexToVertexID[verticesOnTria[3*i+j]]+1);
    	 bulkData->declare_relation(elem, node, j);
     }
    
     int* p_rank = (int*)stk_classic::mesh::field_data(*proc_rank_field, elem);
     p_rank[0] = comm->MyPID();
  }

  for (int i=0; i<indexToEdgeID.size(); i++) {

	 if(isBoundaryEdge[i])
	 {

		 singlePartVec[0] = ssPartVec["lateralside"];
		 stk_classic::mesh::Entity& side = bulkData->declare_entity(metaData->side_rank(), indexToEdgeID[i]+1, singlePartVec);
		 stk_classic::mesh::Entity& elem  = *bulkData->get_entity(metaData->element_rank(),  elem_map->GID(trianglesOnEdge[2*i])+1);
		 bulkData->declare_relation(elem, side,  trianglesPositionsOnEdge[2*i] /*local side id*/);
		 for(int j=0; j<2; j++)
		 {
			 stk_classic::mesh::Entity& node = *bulkData->get_entity(metaData->node_rank(), indexToVertexID[verticesOnEdge[2*i+j]]+1);
			 bulkData->declare_relation(side, node, j);
		 }
	 }
  }

  bulkData->modification_end();
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::MpasSTKMeshStruct::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getValidGenericSTKParameters("Valid ASCII_DiscParams");

  return validPL;
}
