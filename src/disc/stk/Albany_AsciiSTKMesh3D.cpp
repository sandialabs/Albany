//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>

#include "Albany_AsciiSTKMesh3D.hpp"
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

Albany::AsciiSTKMesh3D::AsciiSTKMesh3D(
                                             const Teuchos::RCP<Teuchos::ParameterList>& params, 
                                             const Teuchos::RCP<const Epetra_Comm>& comm) :
  GenericSTKMeshStruct(params,Teuchos::null,3),
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  periodic(false)
{
	/*
   if(comm->MyPID() == 0)
   {
	   int numProc = comm->NumProc(); //total number of processors
	   std::cout << "Number of processors: " << numProc << std::endl;
	   //names of files giving the mesh
	   char meshfilename[100];
	   char shfilename[100];
	   char confilename[100];
	   char bffilename[100];
	   char geIDsfilename[100];
	   char gnIDsfilename[100];
	   char bfIDsfilename[100];
	/*   if (numProc == 1) { //serial run

		 sprintf(shfilename, "%s", "sh");
		 sprintf(confilename, "%s", "eles");
		 sprintf(bffilename, "%s", "bf");
	   }
* /
	    sprintf(meshfilename, "%s", "xyz");
		FILE *meshfile = fopen(meshfilename,"r");
		if (meshfile == NULL) { //check if coordinates file exists
		  *out << "Error in AsciiSTKMesh3D: coordinates file " << meshfilename <<" not found!"<< std::endl;
		  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			  std::endl << "Error in AsciiSTKMesh3D: coordinates file " << meshfilename << " not found!"<< std::endl);
		}
		int temp;
		fseek(meshfile, 0, SEEK_SET);
		char buffer[100];
		fgets(buffer, 100, shfilename);
		sscanf(buffer, "%d, &NumNodes");
		//read in coordinates of mesh -- assumes mesh file is called "xyz" and its first row is the number of nodes
		std::cout << "numNodes: " << NumNodes << std::endl;
		xyz = new double[NumNodes][3];

		for (int i=0; i<NumNodes; i++){
		  fgets(buffer, 100, meshfile);
		  sscanf(buffer, "%lf %lf %lf", &xyz[i][0], &xyz[i][1], &xyz[i][2]);
		  *out << "i: " << i << ", x: " << xyz[i][0] << ", y: " << xyz[i][1] << ", z: " << xyz[i][2] << std::endl;
		}
		eles = new int[NumEles][4];
		//read in connectivity file -- assumes that after the coordinates there is the number of elements
		std::cout << "numEles: " << NumEles << std::endl;
		for (int i=0; i<NumEles; i++){
				  fgets(buffer, 100, meshfile);
				  sscanf(buffer, "%d %d %d %d", &eles[i][0], &eles[i][1], &eles[i][2], &temp);
				  *out << "elm" << i << ": " << eles[i][0] << " " << eles[i][1] << " " << eles[i][2]  << std::endl;
				}

	    //read in lateral edge connectivity file from ascii file
		std::cout << "numBdEdges: " << NumBdEdges << std::endl;
		be = new int[NumBdEdges][2]; //1st column of bf: element # that face belongs to, 2rd-5th columns of bf: connectivity (hard-coded for quad faces)
		for (int i=0; i<NumBdEdges; i++){
		  fgets(buffer, 100, meshfile);
		  sscanf(buffer, "%i %i %i", &be[i][0], &be[i][1], &temp);
		  *out << "edge #:"<<  i << " " << be[i][0] << " " << be[i][1] <<  std::endl;
		}

		fclose(meshfile);
   }
   else
   {
	   NumNodes=NumBdEdges=NumEles=0;
   }
*/
      params->validateParameters(*getValidDiscretizationParameters(),0);

      std::string ebn="Element Block 0";
      partVec[0] = & metaData->declare_part(ebn, metaData->element_rank() );
      ebNameToIndex[ebn] = 0;

    #ifdef ALBANY_SEACAS
      stk::io::put_io_part_attribute(*partVec[0]);
    #endif

      std::vector<std::string> nsNames;
      std::string nsn="Lateral";
      nsNames.push_back(nsn);
      nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
    #ifdef ALBANY_SEACAS
        stk::io::put_io_part_attribute(*nsPartVec[nsn]);
    #endif
      nsn="Internal";
      nsNames.push_back(nsn);
      nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
    #ifdef ALBANY_SEACAS
        stk::io::put_io_part_attribute(*nsPartVec[nsn]);
    #endif
      nsn="Bottom";
      nsNames.push_back(nsn);
      nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
    #ifdef ALBANY_SEACAS
    	stk::io::put_io_part_attribute(*nsPartVec[nsn]);
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
        stk::io::put_io_part_attribute(*ssPartVec[ssnLat]);
        stk::io::put_io_part_attribute(*ssPartVec[ssnBottom]);
        stk::io::put_io_part_attribute(*ssPartVec[ssnTop]);
    #endif

      stk::mesh::fem::set_cell_topology<shards::Tetrahedron<4> >(*partVec[0]);
      stk::mesh::fem::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnBottom]);
      stk::mesh::fem::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnTop]);
      stk::mesh::fem::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnLat]);
      /*
       stk::mesh::fem::set_cell_topology<shards::Wedge<6> >(*partVec[0]);
       stk::mesh::fem::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnBottom]);
       stk::mesh::fem::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssnTop]);
       stk::mesh::fem::set_cell_topology<shards::Quadrilateral<4> >(*ssPartVec[ssnLat]);
*/
      /*
      stk::mesh::fem::set_cell_topology<shards::Hexahedron<8> >(*partVec[0]);
      stk::mesh::fem::set_cell_topology<shards::Quadrilateral<4>>(*ssPartVec[ssnBottom]);
      stk::mesh::fem::set_cell_topology<shards::Quadrilateral<4> >(*ssPartVec[ssnTop]);
      stk::mesh::fem::set_cell_topology<shards::Quadrilateral<4> >(*ssPartVec[ssnLat]);
*/

      meshStruct2D = Teuchos::rcp(new Albany::IossSTKMeshStruct(params, adaptParams, comm));
      Teuchos::RCP<Albany::StateInfoStruct> sis=Teuchos::rcp(new Albany::StateInfoStruct);
      Albany::AbstractFieldContainer::FieldContainerRequirements req;
      meshStruct2D->setFieldAndBulkData(comm, params, 1, req, sis, meshStruct2D->getMeshSpecs()[0]->worksetSize);



		std::vector<stk::mesh::Entity * > cells;
		stk::mesh::Selector select_owned_in_part =
			      stk::mesh::Selector( meshStruct2D->metaData->universal_part() ) &
			      stk::mesh::Selector( meshStruct2D->metaData->locally_owned_part() );
		int numCells = stk::mesh::count_selected_entities( select_owned_in_part, meshStruct2D->bulkData->buckets( meshStruct2D->metaData->element_rank() ));

		numDim = 3;
		int cub = params->get("Cubature Degree",3);
		int worksetSizeMax = params->get("Workset Size",50);
		int worksetSize = this->computeWorksetSize(worksetSizeMax, numCells);

		const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();

		this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub,
								   nsNames, ssNames, worksetSize, partVec[0]->name(),
								   ebNameToIndex, this->interleavedOrdering));

}

Albany::AsciiSTKMesh3D::~AsciiSTKMesh3D()
{
}

void
Albany::AsciiSTKMesh3D::setFieldAndBulkData(
                                               const Teuchos::RCP<const Epetra_Comm>& comm,
                                               const Teuchos::RCP<Teuchos::ParameterList>& params,
                                               const unsigned int neq_,
                                               const AbstractFieldContainer::FieldContainerRequirements& req,
                                               const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                                               const unsigned int worksetSize)
{

	int numLayers=10;
	int numGlobalElements2D = 0;
	int maxGlobalVertices2dId =0;
	int numGlobalVertices2D =0;
	int nGlobalEdges2D =0;
	int Ordering = 0;
	bool isTetra=true;

	  stk::mesh::Selector select_owned_in_part =
	      stk::mesh::Selector( meshStruct2D->metaData->universal_part() ) &
	      stk::mesh::Selector( meshStruct2D->metaData->locally_owned_part() );

	  stk::mesh::Selector select_overlap_in_part =
	      stk::mesh::Selector( meshStruct2D->metaData->universal_part() ) &
	      ( stk::mesh::Selector( meshStruct2D->metaData->locally_owned_part() )
	        | stk::mesh::Selector( meshStruct2D->metaData->globally_shared_part() ) );

	  stk::mesh::Selector select_edges =
	  	      stk::mesh::Selector( *meshStruct2D->metaData->get_part("LateralSide") ) &
	  	      ( stk::mesh::Selector( meshStruct2D->metaData->locally_owned_part() )
	  	        | stk::mesh::Selector( meshStruct2D->metaData->globally_shared_part() ) );

	  std::vector<stk::mesh::Entity * > cells;
	  stk::mesh::get_selected_entities( select_overlap_in_part, meshStruct2D->bulkData->buckets( meshStruct2D->metaData->element_rank() ), cells );

	  std::vector<stk::mesh::Entity * > nodes;
	  stk::mesh::get_selected_entities( select_overlap_in_part, meshStruct2D->bulkData->buckets( meshStruct2D->metaData->node_rank() ), nodes );

	  std::vector<stk::mesh::Entity * > edges;
	  stk::mesh::get_selected_entities( select_edges, meshStruct2D->bulkData->buckets( meshStruct2D->metaData->side_rank() ), edges );


	  int maxOwnedElements2D(0), maxOwnedNodes2D(0), maxOwnedSides2D(0), numOwnedNodes2D(0);
	  for(int i=0; i<cells.size(); i++) maxOwnedElements2D=std::max(maxOwnedElements2D,(int)cells[i]->identifier());
	  for(int i=0; i<nodes.size(); i++) maxOwnedNodes2D=std::max(maxOwnedNodes2D,(int)nodes[i]->identifier());
	  for(int i=0; i<edges.size(); i++) maxOwnedSides2D=std::max(maxOwnedSides2D,(int)edges[i]->identifier());
	  numOwnedNodes2D = stk::mesh::count_selected_entities( select_owned_in_part, meshStruct2D->bulkData->buckets( meshStruct2D->metaData->node_rank() ) );



      comm->MaxAll(&maxOwnedElements2D, &numGlobalElements2D, 1);
      comm->MaxAll(&maxOwnedNodes2D, &maxGlobalVertices2dId, 1);
      comm->MaxAll(&maxOwnedSides2D, &nGlobalEdges2D, 1);
      comm->SumAll(&numOwnedNodes2D, &numGlobalVertices2D, 1);

      //std::cout << "Num Global Elements: " << numGlobalElements2D<< " " << maxGlobalVertices2dId<< " " << nGlobalEdges2D << std::endl;



      std::vector<int> indices(nodes.size()), serialIndices;
      for(int i=0; i<nodes.size(); ++i)
    	  indices[i] = nodes[i]->identifier()-1;

      const Epetra_Map nodes_map(-1, indices.size(), &indices[0], 0, *comm);
      int numMyElements = (comm->MyPID()==0) ? numGlobalVertices2D : 0;
      const Epetra_Map serial_nodes_map(-1, numMyElements, 0, *comm);
      Epetra_Import importOperator(nodes_map, serial_nodes_map);

      Epetra_Vector temp(serial_nodes_map);
      Epetra_Vector sHeightVec(nodes_map);
      Epetra_Vector thickVec(nodes_map);
      Epetra_Vector bFrictionVec(nodes_map);

      std::string fname = "surface_height";
      read2DFileSerial(fname, temp, comm);
      sHeightVec.Import(temp, importOperator, Insert);
      fname = "thickness";
      read2DFileSerial(fname, temp, comm);
      thickVec.Import(temp, importOperator, Insert);
      fname = "basal_friction";
      read2DFileSerial(fname, temp, comm);
      bFrictionVec.Import(temp, importOperator, Insert);

      std::vector<Epetra_Vector>  tempT(numLayers+1, Epetra_Vector(serial_nodes_map));
      std::vector<Epetra_Vector>  temperatureVec(numLayers+1, Epetra_Vector(nodes_map));
      fname = "temperature";
      readFileSerial(fname, tempT, comm);
      for(int i=0; i<numLayers+1; i++)
    	  temperatureVec[i].Import(tempT[i], importOperator, Insert);



	  int elemColumnShift = (Ordering == 1) ? 1 : numGlobalElements2D;
	  int lElemColumnShift = (Ordering == 1) ? 1 : cells.size();
	  int elemLayerShift = (Ordering == 0) ? 1 : numLayers;

	  int vertexColumnShift = (Ordering == 1) ? 1 : maxGlobalVertices2dId;
	  int lVertexColumnShift = (Ordering == 1) ? 1 : nodes.size();
	  int vertexLayerShift = (Ordering == 0) ? 1 : numLayers+1;

	  int edgeColumnShift = (Ordering == 1) ? 1 : nGlobalEdges2D;
	  int lEdgeColumnShift = (Ordering == 1) ? 1 : edges.size();
	  int edgeLayerShift = (Ordering == 0) ? 1 : numLayers;

	  
	  this->SetupFieldData(comm, neq_, req, sis, worksetSize);

	  metaData->commit();

	  bulkData->modification_begin(); // Begin modifying the mesh

	  stk::mesh::PartVector nodePartVec;
	    stk::mesh::PartVector singlePartVec(1);
	    stk::mesh::PartVector emptyPartVec;
	    unsigned int ebNo = 0; //element block #???

	    singlePartVec[0] = nsPartVec["Bottom"];


	    AbstractSTKFieldContainer::IntScalarFieldType* proc_rank_field = fieldContainer->getProcRankField();
	    AbstractSTKFieldContainer::VectorFieldType* coordinates_field = fieldContainer->getCoordinatesField();
	    AbstractSTKFieldContainer::ScalarFieldType* surfaceHeight_field = fieldContainer->getSurfaceHeightField();
	    AbstractSTKFieldContainer::ScalarFieldType* thickness_field = fieldContainer->getThicknessField();
	    AbstractSTKFieldContainer::ScalarFieldType* basal_friction_field = fieldContainer->getBasalFrictionField();
	    AbstractSTKFieldContainer::ScalarFieldType* temperature_field = fieldContainer->getTemperatureField();



	    for(int i=0; i< (numLayers+1)*nodes.size(); i++)
	    {
	  	  int ib = (Ordering == 0)*(i%lVertexColumnShift) + (Ordering == 1)*(i/vertexLayerShift);
	  	  int il = (Ordering == 0)*(i/lVertexColumnShift) + (Ordering == 1)*(i%vertexLayerShift);

	  	  stk::mesh::Entity* node;
	  	  stk::mesh::Entity* node2d = nodes[ib];
	  	  int node2dId = node2d->identifier()-1;
	  	  if(il == 0)
	  		  node = &bulkData->declare_entity(metaData->node_rank(), il*vertexColumnShift+vertexLayerShift * node2dId +1 , singlePartVec);
	  	  else
	  		  node = &bulkData->declare_entity(metaData->node_rank(), il*vertexColumnShift+vertexLayerShift * node2dId +1, nodePartVec);

	  	  int lid = nodes_map.LID(node2dId);
	  	  double* sHeight = stk::mesh::field_data(*surfaceHeight_field, *node);
	  	   sHeight[0] = sHeightVec[lid];

	  	  double* thick = stk::mesh::field_data(*thickness_field, *node);
	  	   thick[0] = thickVec[lid];

	  	  double* bFriction = stk::mesh::field_data(*basal_friction_field, *node);
	  	   bFriction[0] = bFrictionVec[lid];

	  	 double* coord = stk::mesh::field_data(*coordinates_field, *node);
	  	 double* coord2d = stk::mesh::field_data(*coordinates_field, *node2d);

	  	 coord[0] = coord2d[0];   coord[1] = coord2d[1]; coord[2] = sHeight[0] - thick[0] * (1.-double(il)/numLayers);
        }

	    int tetrasLocalIdsOnPrism[3][4];

	      for (int i=0; i<cells.size()*numLayers; i++) {

	    	 int ib = (Ordering == 0)*(i%lElemColumnShift) + (Ordering == 1)*(i/elemLayerShift);
	    	 int il = (Ordering == 0)*(i/lElemColumnShift) + (Ordering == 1)*(i%elemLayerShift);

	    	 int shift = il*vertexColumnShift;

	    	 singlePartVec[0] = partVec[ebNo];


	         //TODO: this could be done only in the first layer and then copied into the other layers
	         int prismMpasIds[3], prismGlobalIds[6];
	         stk::mesh::PairIterRelation rel = cells[ib]->relations(meshStruct2D->metaData->node_rank());
	         double tempOnPrism=0; //Set temperature constant on each prism/Hexa
	         for (int j = 0; j < 3; j++)
	    	 {
	        	 int node2dId = rel[j].entity()->identifier()-1;
	        	 int node2dLId = nodes_map.LID(node2dId);
	        	 int mpasLowerId = vertexLayerShift * node2dId;
	        	 int lowerId = shift+vertexLayerShift * node2dId;
	        	 prismMpasIds[j] = mpasLowerId;
	    		 prismGlobalIds[j] = lowerId;
	    		 prismGlobalIds[j + 3] = lowerId+vertexColumnShift;
	    		 tempOnPrism += 1./6.*(temperatureVec[il][node2dLId]+temperatureVec[il+1][node2dLId]);

	    	 }

	         tetrasFromPrismStructured (prismMpasIds, prismGlobalIds, tetrasLocalIdsOnPrism);

	         int prismId = il*elemColumnShift+ elemLayerShift * (cells[ib]->identifier()-1);
	         for(int iTetra = 0; iTetra<3; iTetra++)
	         {
	        	 stk::mesh::Entity& elem  = bulkData->declare_entity(metaData->element_rank(), 3*prismId+iTetra+1, singlePartVec);
	    		 for(int j=0; j<4; j++)
	    		 {
	    			 stk::mesh::Entity& node = *bulkData->get_entity(metaData->node_rank(), tetrasLocalIdsOnPrism[iTetra][j]+1);
	    			 bulkData->declare_relation(elem, node, j);
	    		 }
	    		 int* p_rank = (int*)stk::mesh::field_data(*proc_rank_field, elem);
	    		 p_rank[0] = comm->MyPID();
	    		 double* temperature = stk::mesh::field_data(*temperature_field, elem);
	    		 temperature[0] = tempOnPrism;
	         }
	      }

	      singlePartVec[0] = ssPartVec["lateralside"];

	      //first we store the lateral faces of prisms, which corresponds to edges of the basal mesh
	      int tetraSidePoints[4][3] = {{0, 1, 3}, {1, 2, 3}, {0, 3, 2}, {0, 2, 1}};
	      std::vector<int> tetraPos(2), facePos(2);

	      std::vector<std::vector<std::vector<int> > > prismStruct(3, std::vector<std::vector<int> >(4, std::vector<int>(3)));
	      for (int i=0; i<edges.size()*numLayers; i++) {
	    	 int ib = (Ordering == 0)*(i%lEdgeColumnShift) + (Ordering == 1)*(i/edgeLayerShift);
	    	// if(isBoundaryEdge[ib])
	    	 {
	    		 stk::mesh::Entity* edge2d = edges[ib];
	    		 stk::mesh::PairIterRelation rel = edge2d->relations(meshStruct2D->metaData->element_rank());
	    		 int il = (Ordering == 0)*(i/lEdgeColumnShift) + (Ordering == 1)*(i%edgeLayerShift);
	    		 stk::mesh::Entity* elem2d = rel[0].entity();

	    		 int basalElemId = elem2d->identifier()-1;
	    		 int Edge2dId = edge2d->identifier()-1;

	    		 //TODO: this could be done only in the first layer and then copied into the other layers
	    		 int prismMpasIds[3], prismGlobalIds[6];
	    		 int shift = il*vertexColumnShift;
	    		 rel = elem2d->relations(meshStruct2D->metaData->node_rank());
	    		 for (int j = 0; j < 3; j++)
	    		 {
	    			 int node2dId = rel[j].entity()->identifier()-1;
	    			 int mpasLowerId = vertexLayerShift * node2dId;
	    			 int lowerId = shift+vertexLayerShift * node2dId;
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


	    		 rel = edge2d->relations(meshStruct2D->metaData->node_rank());
	    		 int node2dId_0 = rel[0].entity()->identifier()-1;
	    	     int node2dId_1 = rel[1].entity()->identifier()-1;
	    		 int basalVertexId[2] = {node2dId_0*vertexLayerShift, node2dId_1*vertexLayerShift};
	    		 std::vector<int> bdPrismFaceIds(4);

	    		 bdPrismFaceIds[0] = node2dId_0*vertexLayerShift+vertexColumnShift*il+1;
	    		 bdPrismFaceIds[1] = node2dId_1*vertexLayerShift+vertexColumnShift*il+1;
	    		 bdPrismFaceIds[2] = bdPrismFaceIds[0]+vertexColumnShift;
	    		 bdPrismFaceIds[3] = bdPrismFaceIds[1]+vertexColumnShift;

	    		 setBdFacesOnPrism (prismStruct, bdPrismFaceIds, tetraPos, facePos);

	    		 int basalEdgeId = Edge2dId*2*edgeLayerShift;
	    		 for(int k=0; k< tetraPos.size(); k++)
	    		 {
	    			 int iTetra = tetraPos[k];
	    			 int iFace = facePos[k];
	    			 stk::mesh::Entity& elem = *bulkData->get_entity(metaData->element_rank(), 3*il*elemColumnShift+ 3*elemLayerShift * basalElemId +iTetra+1);
	    			 std::vector<int>& faceIds = prismStruct[iTetra][iFace];
	    			 stk::mesh::Entity& side = bulkData->declare_entity(metaData->side_rank(), 2*edgeColumnShift*il+basalEdgeId+k+1, singlePartVec);
	    			// if(edgeColumnShift*il+basalEdgeId+k+1==133) throw;
	    			 bulkData->declare_relation(elem, side,  iFace );
	    			 for(int j=0; j<3; j++)
	    			 {
	    				 stk::mesh::Entity& node = *bulkData->get_entity(metaData->node_rank(), faceIds[j]);
	    				 bulkData->declare_relation(side, node, j);
	    			 }
	    		 }
	    	 }
	      }



	      //then we store the lower and upper faces of prisms, which corresponds to triangles of the basal mesh

		edgeLayerShift = (Ordering == 0) ? 1 : numLayers+1;
		edgeColumnShift = elemColumnShift;

		singlePartVec[0] = ssPartVec["basalside"];

		int edgeOffset = 2*nGlobalEdges2D*numLayers;
		for (int i=0; i<cells.size(); i++)
		{
		  stk::mesh::Entity& elem2d = *cells[i];
		  int elem2d_id = elem2d.identifier()-1;
		  stk::mesh::Entity& side = bulkData->declare_entity(metaData->side_rank(), elem2d_id*edgeLayerShift+edgeOffset+1, singlePartVec);
		  stk::mesh::Entity& elem  = *bulkData->get_entity(metaData->element_rank(),  elem2d_id*3*elemLayerShift+1);
		  bulkData->declare_relation(elem, side,  3);
		  stk::mesh::PairIterRelation rel = elem2d.relations(meshStruct2D->metaData->node_rank());
		  for(int j=0; j<3; j++)
		  {
			 int node2dId = rel[j].entity()->identifier()-1;
			 stk::mesh::Entity& node = *bulkData->get_entity(metaData->node_rank(), vertexLayerShift*node2dId+1);
			 bulkData->declare_relation(side, node, j);
		  }
		}

		singlePartVec[0] = ssPartVec["upperside"];

		for (int i=0; i<cells.size(); i++)
		{
			stk::mesh::Entity& elem2d = *cells[i];
		    int elem2d_id = elem2d.identifier()-1;
			stk::mesh::Entity& side = bulkData->declare_entity(metaData->side_rank(), elem2d_id*edgeLayerShift+numLayers*2*edgeColumnShift+edgeOffset+1, singlePartVec);
			stk::mesh::Entity& elem  = *bulkData->get_entity(metaData->element_rank(),  elem2d_id*3*elemLayerShift+(numLayers-1)*3*elemColumnShift+1+2);
			bulkData->declare_relation(elem, side,  1);
			stk::mesh::PairIterRelation rel = elem2d.relations(meshStruct2D->metaData->node_rank());
			for(int j=0; j<3; j++)
			{
			  int node2dId = rel[j].entity()->identifier()-1;
			  stk::mesh::Entity& node = *bulkData->get_entity(metaData->node_rank(), vertexLayerShift*node2dId+numLayers*vertexColumnShift+1);
			  bulkData->declare_relation(side, node, j);
			}
		}



/*
	    for (int i=0; i<cells.size()*numLayers; i++) {

	    	 int ib = (Ordering == 0)*(i%lElemColumnShift) + (Ordering == 1)*(i/elemLayerShift);
	    	 int il = (Ordering == 0)*(i/lElemColumnShift) + (Ordering == 1)*(i%elemLayerShift);

	    	 int shift = il*vertexColumnShift;

	    	 singlePartVec[0] = partVec[ebNo];
	    	 int elemId = il*elemColumnShift+elemLayerShift * (cells[ib]->identifier()-1) +1;
	         stk::mesh::Entity& elem  = bulkData->declare_entity(metaData->element_rank(),elemId, singlePartVec);

	         stk::mesh::PairIterRelation rel = cells[ib]->relations(metaData->node_rank());
	         for(int j=0; j<3; j++)
	         {

	        	 int node2dId = rel[j].entity()->identifier()-1;
	        	 int lowerId = shift+vertexLayerShift * node2dId+1;
	        	 stk::mesh::Entity& node = *bulkData->get_entity(metaData->node_rank(), lowerId);
	        	 bulkData->declare_relation(elem, node, j);

	        	 stk::mesh::Entity& node_top = *bulkData->get_entity(metaData->node_rank(), lowerId+vertexColumnShift);
	        	 bulkData->declare_relation(elem, node_top, j+3);
	         }

	         int* p_rank = (int*)stk::mesh::field_data(*proc_rank_field, elem);
	         p_rank[0] = comm->MyPID();
	      }
*/

	  bulkData->modification_end();

}


Teuchos::RCP<const Teuchos::ParameterList>
Albany::AsciiSTKMesh3D::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getValidGenericSTKParameters("Valid ASCII_DiscParams");
    validPL->set<std::string>("Exodus Input File Name", "", "File Name For Exodus Mesh Input");

  return validPL;
}

  void Albany::AsciiSTKMesh3D::read2DFileSerial(std::string &fname, Epetra_Vector& content, const Teuchos::RCP<const Epetra_Comm>& comm)
  {
	  int numNodes;
	  if(comm->MyPID()==0)
	  {
		  std::ifstream ifile;
		  ifile.open (fname.c_str() );
		  if (ifile.is_open() )
		  {
			  ifile >> numNodes;
			  TEUCHOS_ASSERT(numNodes == content.MyLength());
			  for (int i = 0; i < numNodes; i++)
				  ifile >> content[i];
			  ifile.close();
		   }
		   else
		   {
			  std::cout << "Unable to open the file " << fname << std::endl;
		   }
	  }
  }

  void Albany::AsciiSTKMesh3D::readFileSerial(std::string &fname, std::vector<Epetra_Vector>& contentVec, const Teuchos::RCP<const Epetra_Comm>& comm)
  {
  	  int numNodes, numComponents;
  	  if(comm->MyPID()==0)
  	  {
  		  std::ifstream ifile;
  		  ifile.open (fname.c_str() );
  		  if (ifile.is_open() )
  		  {
  			  ifile >> numNodes >> numComponents;
  			  TEUCHOS_ASSERT(numNodes == contentVec[0].MyLength());
  			  TEUCHOS_ASSERT(numComponents == contentVec.size());
  			  for(int il=0;il<numComponents; ++il)
  			    for (int i = 0; i < numNodes; i++)
  				  ifile >> contentVec[il][i];
  			  ifile.close();
  		   }
  		   else
  		   {
  			  std::cout << "Unable to open the file " << fname << std::endl;
  		   }
  	  }
  }

