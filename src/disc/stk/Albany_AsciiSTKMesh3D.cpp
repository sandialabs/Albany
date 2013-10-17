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
  periodic(false),
  sh(0)
{
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
*/
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
		fgets(buffer, 100, meshfile);
		sscanf(buffer, "%d %d %d", &NumNodes, &NumEles, &NumBdEdges);
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

      params->validateParameters(*getValidDiscretizationParameters(),0);

      std::string ebn="Element Block 0";
      partVec[0] = & metaData->declare_part(ebn, metaData->element_rank() );
      ebNameToIndex[ebn] = 0;

    #ifdef ALBANY_SEACAS
    //  stk::io::put_io_part_attribute(metaData->universal_part());
      stk::io::put_io_part_attribute(*partVec[0]);
    #endif

    /*  std::vector<std::string> nsNames;
      std::string nsn="Lateral";
      nsNames.push_back(nsn);
      nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
    #ifdef ALBANY_SEACAS
        stk::io::put_io_part_attribute(*nsPartVec[nsn]);
    #endif
    */
      std::vector<std::string> nsNames;
      std::string nsn="Node";
      nsNames.push_back(nsn);
      nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
    #ifdef ALBANY_SEACAS
        stk::io::put_io_part_attribute(*nsPartVec[nsn]);
    #endif

      std::vector<std::string> ssNames;
      std::string ssn="LateralSide";
      ssNames.push_back(ssn);
        ssPartVec[ssn] = & metaData->declare_part(ssn, metaData->side_rank() );
    #ifdef ALBANY_SEACAS
        stk::io::put_io_part_attribute(*ssPartVec[ssn]);
    #endif

      stk::mesh::fem::set_cell_topology<shards::Tetrahedron<4> >(*partVec[0]);
      stk::mesh::fem::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssn]);
      numDim = 2;
      int cub = params->get("Cubature Degree",3);
      int worksetSizeMax = params->get("Workset Size",50);
      int worksetSize = this->computeWorksetSize(worksetSizeMax, NumEles);
      *out << __LINE__ <<std::endl;
      const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();
      cullSubsetParts(ssNames, ssPartVec);
      this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub,
                                 nsNames, ssNames, worksetSize, partVec[0]->name(),
                                 ebNameToIndex, this->interleavedOrdering));

      meshStruct2D = Teuchos::rcp(new Albany::IossSTKMeshStruct(params, adaptParams, comm));
      Teuchos::RCP<Albany::StateInfoStruct> sis=Teuchos::rcp(new Albany::StateInfoStruct);
      Albany::AbstractFieldContainer::FieldContainerRequirements req;
      meshStruct2D->setFieldAndBulkData(comm, params, 1, req, sis, meshStruct2D->getMeshSpecs()[0]->worksetSize);


     std::cout<< "Spatial dim: " << metaData->spatial_dimension() <<std::endl;
}

Albany::AsciiSTKMesh3D::~AsciiSTKMesh3D()
{
  delete [] xyz; 
  delete [] be;
  delete [] eles; 
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
	int numGlobalElements2D = 4;
	int nGlobalVertices2D =6;
	int nGlobalEdges2D =0;
	int Ordering = 0;



	  Albany::STKDiscretization disc2D(meshStruct2D, comm);
	  Teuchos::RCP<const Epetra_Map> overlapMap2D = disc2D.getOverlapMap();

	  stk::mesh::Selector select_owned_in_part =
	      stk::mesh::Selector( meshStruct2D->metaData->universal_part() ) &
	      stk::mesh::Selector( meshStruct2D->metaData->locally_owned_part() );

	  stk::mesh::Selector select_overlap_in_part =
	      stk::mesh::Selector( meshStruct2D->metaData->universal_part() ) &
	      ( stk::mesh::Selector( meshStruct2D->metaData->locally_owned_part() )
	        | stk::mesh::Selector( meshStruct2D->metaData->globally_shared_part() ) );


	  std::vector<stk::mesh::Entity * > cells;
	  stk::mesh::get_selected_entities( select_owned_in_part, meshStruct2D->bulkData->buckets( meshStruct2D->metaData->element_rank() ), cells );

	  std::vector<stk::mesh::Entity * > nodes;
	  stk::mesh::get_selected_entities( select_overlap_in_part, meshStruct2D->bulkData->buckets( meshStruct2D->metaData->node_rank() ), nodes );

	  std::vector<stk::mesh::Entity * > edges;
	  stk::mesh::get_selected_entities( select_overlap_in_part, meshStruct2D->bulkData->buckets( meshStruct2D->metaData->node_rank() ), edges );

	  int elemColumnShift = (Ordering == 1) ? 1 : numGlobalElements2D;
	  int lElemColumnShift = (Ordering == 1) ? 1 : cells.size();
	  	    int elemLayerShift = (Ordering == 0) ? 1 : numLayers;

	  	    int vertexColumnShift = (Ordering == 1) ? 1 : nGlobalVertices2D;
	  	    int lVertexColumnShift = (Ordering == 1) ? 1 : nodes.size();
	  	    int vertexLayerShift = (Ordering == 0) ? 1 : numLayers+1;

	  	    int edgeColumnShift = (Ordering == 1) ? 1 : nGlobalEdges2D;
	  	    int lEdgeColumnShift = (Ordering == 1) ? 1 : edges.size();
	  	    int edgeLayerShift = (Ordering == 0) ? 1 : numLayers;



	  for(int i=0; i<cells.size();i++)
		  std::cout << cells[i]->identifier() << ", ";
	  std::cout <<std::endl;

	  for(int i=0; i<nodes.size();i++)
	  		  std::cout << nodes[i]->identifier() << ",, ";
	  	  std::cout <<std::endl;


	  for(int i=0; i<overlapMap2D->NumMyElements();i++)
		  std::cout << overlapMap2D->MyGlobalElements()[i] << " ";
	  std::cout <<std::endl;
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
	//  	  int numBdEdges(0);
	//  	  for (int i=0; i<indexToEdgeID.size(); i++)
	//  		  numBdEdges += isBoundaryEdge[i];


	        double* coord = stk::mesh::field_data(*coordinates_field, *node);
	        double* coord2d = stk::mesh::field_data(*coordinates_field, *node2d);

	  	  coord[0] = coord2d[0];   coord[1] = coord2d[1]; coord[2] = double(il)/numLayers;

	  	  double* sHeight;
	  	   sHeight = stk::mesh::field_data(*surfaceHeight_field, *node);
	  	   sHeight[0] = 1.;
	    }

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
