//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>

#include "Albany_AsciiSTKMesh2D.hpp"
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

Albany::AsciiSTKMesh2D::AsciiSTKMesh2D(
                                             const Teuchos::RCP<Teuchos::ParameterList>& params, 
                                             const Teuchos::RCP<const Epetra_Comm>& comm) :
  GenericSTKMeshStruct(params,Teuchos::null,2),
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
		  *out << "Error in AsciiSTKMesh2D: coordinates file " << meshfilename <<" not found!"<< std::endl;
		  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			  std::endl << "Error in AsciiSTKMesh2D: coordinates file " << meshfilename << " not found!"<< std::endl);
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
        stk::io::put_io_part_attribute(metaData->universal_part());
    #endif

      stk::mesh::fem::set_cell_topology<shards::Triangle<3> >(*partVec[0]);
      stk::mesh::fem::set_cell_topology<shards::Line<2> >(*ssPartVec[ssn]);
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
     std::cout<< "Spatial dim: " << metaData->spatial_dimension() <<std::endl;
}

Albany::AsciiSTKMesh2D::~AsciiSTKMesh2D()
{
  delete [] xyz; 
  delete [] be;
  delete [] eles; 
}

void
Albany::AsciiSTKMesh2D::setFieldAndBulkData(
                                               const Teuchos::RCP<const Epetra_Comm>& comm,
                                               const Teuchos::RCP<Teuchos::ParameterList>& params,
                                               const unsigned int neq_,
                                               const AbstractFieldContainer::FieldContainerRequirements& req,
                                               const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                                               const unsigned int worksetSize)
{
	  this->SetupFieldData(comm, neq_, req, sis, worksetSize);

	  metaData->commit();

	  bulkData->modification_begin(); // Begin modifying the mesh

	  stk::mesh::PartVector nodePartVec;
	  stk::mesh::PartVector singlePartVec(1);
	  std::cout << "elem_map # elments: " << NumEles << std::endl;
	  unsigned int ebNo = 0; //element block #???
	  int sideID = 0;

	  AbstractSTKFieldContainer::IntScalarFieldType* proc_rank_field = fieldContainer->getProcRankField();
	  AbstractSTKFieldContainer::VectorFieldType* coordinates_field = fieldContainer->getCoordinatesField();
	//  AbstractSTKFieldContainer::ScalarFieldType* surfaceHeight_field = fieldContainer->getSurfaceHeightField();


	  singlePartVec[0] = nsPartVec["Node"];

	  for (int i=0; i<NumNodes; i++)
	  {
		  stk::mesh::Entity& node = bulkData->declare_entity(metaData->node_rank(), i+1, singlePartVec);

		  double* coord;
		  coord = stk::mesh::field_data(*coordinates_field, node);
		  coord[0] = xyz[i][0];   coord[1] = xyz[i][1]; coord[2] = 0.;//xyz[i][2];
	  }

	  for (int i=0; i<NumEles; i++)
	  {

	     singlePartVec[0] = partVec[ebNo];
	     stk::mesh::Entity& elem  = bulkData->declare_entity(metaData->element_rank(), i+1, singlePartVec);

	     for(int j=0; j<3; j++)
	     {
	    	 stk::mesh::Entity& node = *bulkData->get_entity(metaData->node_rank(), eles[i][j]);
	    	 bulkData->declare_relation(elem, node, j);
	     }

	   //  int* p_rank = stk::mesh::field_data(*proc_rank_field, elem);
	 //    p_rank[0] = comm->MyPID();
	  }

	  std::map<std::pair<int,int>, int> edgeMap;
	  for (int i=0; i<NumBdEdges; i++)
		  edgeMap.insert(std::pair<std::pair<int,int>, int>(std::make_pair(be[i][0],be[i][1]),i+1));

	  singlePartVec[0] = ssPartVec["LateralSide"];
	  for (int i=0; i<NumEles; i++)
	  {
		  for(int j=0; j<3; j++)
		  {
			  std::map<std::pair<int,int>, int>::iterator it=edgeMap.find(std::make_pair(eles[i][j],eles[i][(j+1)%3]));
			  if (it != edgeMap.end())
			  {
				  stk::mesh::Entity& side = bulkData->declare_entity(metaData->side_rank(), it->second, singlePartVec);
				  stk::mesh::Entity& elem  = *bulkData->get_entity(metaData->element_rank(),  i+1);
				  bulkData->declare_relation(elem, side,  j /*local side id*/);
				  stk::mesh::Entity& node1 = *bulkData->get_entity(metaData->node_rank(), it->first.first);
				  bulkData->declare_relation(side, node1, 0);
				  stk::mesh::Entity& node2 = *bulkData->get_entity(metaData->node_rank(), it->first.second);
				  bulkData->declare_relation(side, node2, 1);
				  edgeMap.erase(it);
			  }
		  }
	  }

	  bulkData->modification_end();
/*

	  numDim = metaData->spatial_dimension();

	  //  stk::io::put_io_part_attribute(metaData->universal_part());

	    // Set element blocks, side sets and node sets
	    const stk::mesh::PartVector & all_parts = metaData->get_parts();
	    std::vector<std::string> ssNames;
	    std::vector<std::string> nsNames;
	    int numEB = 0;

	    for (stk::mesh::PartVector::const_iterator i = all_parts.begin();
	         i != all_parts.end(); ++i) {

	      stk::mesh::Part * const part = *i ;

	      if ( part->primary_entity_rank() == metaData->element_rank()) {
	        if (part->name()[0] != '{') {
	          //*out << "IOSS-STK: Element part \"" << part->name() << "\" found " << std::endl;
	          partVec[numEB] = part;
	          numEB++;
	        }
	      }
	      else if ( part->primary_entity_rank() == metaData->node_rank()) {
	        if (part->name()[0] != '{') {
	          //*out << "Mesh has Node Set ID: " << part->name() << std::endl;
	          nsPartVec[part->name()]=part;
	          nsNames.push_back(part->name());
	        }
	      }
	      else if ( part->primary_entity_rank() == metaData->side_rank()) {
	        if (part->name()[0] != '{') {
	  //        print(*out, "Found side_rank entity:\n", *part);
	          ssPartVec[part->name()]=part;
	        }
	      }
	    }

	    cullSubsetParts(ssNames, ssPartVec); // Eliminate sidesets that are subsets of other sidesets

*/

#ifdef ALBANY_ZOLTAN

      // Refine the mesh before starting the simulation if indicated
    //  uniformRefineMesh(comm);

      // Rebalance the mesh before starting the simulation if indicated
   //   rebalanceInitialMesh(comm);

#endif


}


Teuchos::RCP<const Teuchos::ParameterList>
Albany::AsciiSTKMesh2D::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getValidGenericSTKParameters("Valid ASCII_DiscParams");

  return validPL;
}
