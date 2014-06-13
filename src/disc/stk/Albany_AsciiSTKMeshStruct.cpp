//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>

#include "Albany_AsciiSTKMeshStruct.hpp"
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


//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN


//Constructor for meshes read from ASCII file 
Albany::AsciiSTKMeshStruct::AsciiSTKMeshStruct(
                                             const Teuchos::RCP<Teuchos::ParameterList>& params, 
                                             const Teuchos::RCP<const Epetra_Comm>& comm) :
  GenericSTKMeshStruct(params,Teuchos::null,3),
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  periodic(false)
{
   int numProc = comm->NumProc(); //total number of processors
   contigIDs = params->get("Contiguous IDs", true); 
   std::cout << "Number of processors: " << numProc << std::endl; 
   //names of files giving the mesh
   char meshfilename[100]; 
   char shfilename[100];
   char confilename[100];
   char bffilename[100];
   char geIDsfilename[100];
   char gnIDsfilename[100];
   char bfIDsfilename[100];
   char flwafilename[100]; //flow factor file
   char tempfilename[100]; //temperature file
   char betafilename[100]; //basal friction coefficient file
   if ((numProc == 1) & (contigIDs == true)) { //serial run with contiguous global IDs
#ifdef OUTPUT_TO_SCREEN
     std::cout << "Ascii mesh has contiguous IDs; no bfIDs, geIDs, gnIDs files required." << std::endl;
#endif
     sprintf(meshfilename, "%s", "xyz");
     sprintf(shfilename, "%s", "sh");
     sprintf(confilename, "%s", "eles");
     sprintf(bffilename, "%s", "bf");
     sprintf(flwafilename, "%s", "flwa");
     sprintf(tempfilename, "%s", "temp");
     sprintf(betafilename, "%s", "beta");
   }
   else { //parallel run or serial run with non-contiguous global IDs - proc # is appended to file name to indicate what processor the mesh piece is on 
#ifdef OUTPUT_TO_SCREEN
     if ((numProc == 1) & (contigIDs == false))
        std::cout << "1 processor run with non-contiguous IDs; bfIDs0, geIDs0, gnIDs0 files required." << std::endl;
#endif
     int suffix = comm->MyPID(); //current processor number 
     sprintf(meshfilename, "%s%i", "xyz", suffix);
     sprintf(shfilename, "%s%i", "sh", suffix);
     sprintf(confilename, "%s%i", "eles", suffix);
     sprintf(bffilename, "%s%i", "bf", suffix);
     sprintf(geIDsfilename, "%s%i", "geIDs", suffix);
     sprintf(gnIDsfilename, "%s%i", "gnIDs", suffix);
     sprintf(bfIDsfilename, "%s%i", "bfIDs", suffix);
     sprintf(flwafilename, "%s%i", "flwa", suffix);
     sprintf(tempfilename, "%s%i", "temp", suffix);
     sprintf(betafilename, "%s%i", "beta", suffix);
   }

    //read in coordinates of mesh -- right now hard coded for 3D
    //assumes mesh file is called "xyz" and its first row is the number of nodes  
    FILE *meshfile = fopen(meshfilename,"r");
    if (meshfile == NULL) { //check if coordinates file exists
      *out << "Error in AsciiSTKMeshStruct: coordinates file " << meshfilename <<" not found!"<< std::endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
          std::endl << "Error in AsciiSTKMeshStruct: coordinates file " << meshfilename << " not found!"<< std::endl);
    }
    double temp; 
    fseek(meshfile, 0, SEEK_SET); 
    fscanf(meshfile, "%lf", &temp); 
    NumNodes = int(temp); 
#ifdef OUTPUT_TO_SCREEN
    *out << "numNodes: " << NumNodes << std::endl;  
#endif
    xyz = new double[NumNodes][3]; 
    char buffer[100];
    fgets(buffer, 100, meshfile); 
    for (int i=0; i<NumNodes; i++){
      fgets(buffer, 100, meshfile); 
      sscanf(buffer, "%lf %lf %lf", &xyz[i][0], &xyz[i][1], &xyz[i][2]); 
      //*out << "i: " << i << ", x: " << xyz[i][0] << ", y: " << xyz[i][1] << ", z: " << xyz[i][2] << std::endl; 
     }
    //read in surface height data from mesh 
    //assumes surface height file is called "sh" and its first row is the number of nodes  
    FILE *shfile = fopen(shfilename,"r");
    have_sh = false;
    if (shfile != NULL) have_sh = true;
    if (have_sh) {
      fseek(shfile, 0, SEEK_SET); 
      fscanf(shfile, "%lf", &temp); 
      int NumNodesSh = int(temp);
#ifdef OUTPUT_TO_SCREEN
      *out << "NumNodesSh: " << NumNodesSh<< std::endl;
#endif 
      if (NumNodesSh != NumNodes) { 
           *out << "Error in AsciiSTKMeshStruct: sh file must have same number nodes as xyz file!  numNodes in xyz = " << NumNodes <<", numNodes in sh = "<< NumNodesSh  << std::endl;
          TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            std::endl << "Error in AsciiSTKMeshStruct: sh file must have same number nodes as xyz file!  numNodes in xyz = " << NumNodes << ", numNodes in sh = "<< NumNodesSh << std::endl);
      }
      sh = new double[NumNodes]; 
      fgets(buffer, 100, shfile); 
      for (int i=0; i<NumNodes; i++){
        fgets(buffer, 100, shfile); 
        sscanf(buffer, "%lf", &sh[i]); 
        //*out << "i: " << i << ", sh: " << sh[i] << std::endl; 
       }
     }
     //read in connectivity file -- right now hard coded for 3D hexes
     //assumes mesh file is called "eles" and its first row is the number of elements  
     FILE *confile = fopen(confilename,"r"); 
     if (confile == NULL) { //check if element connectivity file exists
      *out << "Error in AsciiSTKMeshStruct: element connectivity file " << confilename <<" not found!"<< std::endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
          std::endl << "Error in AsciiSTKMeshStruct: element connectivity file " << confilename << " not found!"<< std::endl);
     }
     fseek(confile, 0, SEEK_SET); 
     fscanf(confile, "%lf", &temp); 
     NumEles = int(temp); 
#ifdef OUTPUT_TO_SCREEN
     *out << "numEles: " << NumEles << std::endl;
#endif 
     eles = new int[NumEles][8]; 
     fgets(buffer, 100, confile); 
     for (int i=0; i<NumEles; i++){
        fgets(buffer, 100, confile); 
        sscanf(buffer, "%i %i %i %i %i %i %i %i", &eles[i][0], &eles[i][1], &eles[i][2], &eles[i][3], &eles[i][4], &eles[i][5], &eles[i][6], &eles[i][7]);
        //*out << "elt # " << i << ": " << eles[i][0] << " " << eles[i][1] << " " << eles[i][2] << " " << eles[i][3] << " " << eles[i][4] << " "
        //                  << eles[i][5] << " " << eles[i][6] << " " << eles[i][7] << std::endl; 
     }
    //read in basal face connectivity file from ascii file
    //assumes basal face connectivity file is called "bf" and its first row is the number of faces on basal boundary
    FILE *bffile = fopen(bffilename,"r");
    have_bf = false;
    if (bffile != NULL) have_bf = true;
    if (have_bf) {
      fseek(bffile, 0, SEEK_SET); 
      fscanf(bffile, "%lf", &temp); 
      NumBasalFaces = int(temp); 
#ifdef OUTPUT_TO_SCREEN
      *out << "numBasalFaces: " << NumBasalFaces << std::endl;  
#endif
      bf = new int[NumBasalFaces][5]; //1st column of bf: element # that face belongs to, 2rd-5th columns of bf: connectivity (hard-coded for quad faces) 
      fgets(buffer, 100, bffile); 
      for (int i=0; i<NumBasalFaces; i++){
        fgets(buffer, 100, bffile); 
        sscanf(buffer, "%i %i %i %i %i", &bf[i][0], &bf[i][1], &bf[i][2], &bf[i][3], &bf[i][4]); 
        //*out << "face #:" << bf[i][0] << ", face conn:" << bf[i][1] << " " << bf[i][2] << " " << bf[i][3] << " " << bf[i][4] << std::endl; 
       }
     }
     //Create array w/ global element IDs 
     globalElesID = new int[NumEles];
     if ((numProc == 1) & (contigIDs == true)) { //serial run with contiguous global IDs: element IDs are just 0->NumEles-1
       for (int i=0; i<NumEles; i++) {
          globalElesID[i] = i; 
          //*out << "local element ID #:" << i << ", global element ID #:" << globalElesID[i] << std::endl;
       }
     }
     else {//parallel run: read global element IDs from file.  
           //This file should have a header like the other files, and length NumEles.
       FILE *geIDsfile = fopen(geIDsfilename,"r");
       if (geIDsfile == NULL) { //check if global element IDs file exists
         *out << "Error in AsciiSTKMeshStruct: global element IDs file " << geIDsfilename <<" not found!"<< std::endl;
         TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            std::endl << "Error in AsciiSTKMeshStruct: global element IDs file " << geIDsfilename << " not found!"<< std::endl);
       }
       fseek(geIDsfile, 0, SEEK_SET);
       fgets(buffer, 100, geIDsfile);
       for (int i=0; i<NumEles; i++){
         fgets(buffer, 100, geIDsfile);
         sscanf(buffer, "%i ", &globalElesID[i]);
         globalElesID[i] = globalElesID[i]-1; //subtract 1 b/c global element IDs file assumed to be 1-based not 0-based
         //*out << "local element ID #:" << i << ", global element ID #:" << globalElesID[i] << std::endl;
       }
     }
     //Create array w/ global node IDs 
     globalNodesID = new int[NumNodes];
     if ((numProc == 1) & (contigIDs == true)) { //serial run with contiguous global IDs: element IDs are just 0->NumEles-1
       for (int i=0; i<NumNodes; i++) { 
          globalNodesID[i] = i; 
          //*out << "local node ID #:" << i << ", global node ID #:" << globalNodesID[i] << std::endl;
       }
     }
     else {//parallel run: read global node IDs from file.  
           //This file should have a header like the other files, and length NumNodes
       FILE *gnIDsfile = fopen(gnIDsfilename,"r");
       if (gnIDsfile == NULL) { //check if global node IDs file exists
         *out << "Error in AsciiSTKMeshStruct: global node IDs file " << gnIDsfilename <<" not found!"<< std::endl;
         TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            std::endl << "Error in AsciiSTKMeshStruct: global node IDs file " << gnIDsfilename << " not found!"<< std::endl);
       }
       fseek(gnIDsfile, 0, SEEK_SET);
       fgets(buffer, 100, gnIDsfile);
       for (int i=0; i<NumNodes; i++){
         fgets(buffer, 100, gnIDsfile);
         sscanf(buffer, "%i ", &globalNodesID[i]);
         globalNodesID[i] = globalNodesID[i]-1; //subtract 1 b/c global node IDs file assumed to be 1-based not 0-based 
         //*out << "local node ID #:" << i << ", global node ID #:" << globalNodesID[i] << std::endl;
       }
     }
     basalFacesID = new int[NumBasalFaces];
     if ((numProc == 1) & (contigIDs == true)) { //serial run with contiguous global IDs: element IDs are just 0->NumEles-1
       for (int i=0; i<NumBasalFaces; i++) { 
          basalFacesID[i] = i; 
          //*out << "local face ID #:" << i << ", global face ID #:" << basalFacesID[i] << std::endl;
       }
     }
     else {//parallel run: read basal face IDs from file.  
           //This file should have a header like the other files, and length NumBasalFaces
       FILE *bfIDsfile = fopen(bfIDsfilename,"r");
       fseek(bfIDsfile, 0, SEEK_SET);
       fgets(buffer, 100, bfIDsfile);
       for (int i=0; i<NumBasalFaces; i++){
         fgets(buffer, 100, bfIDsfile);
         sscanf(buffer, "%i ", &basalFacesID[i]);
         basalFacesID[i] = basalFacesID[i]-1; //subtract 1 b/c basal face IDs file assumed to be 1-based not 0-based
         //*out << "local face ID #:" << i << ", global face ID #:" << basalFacesID[i] << std::endl;
       }
     }
    //read in flow factor (flwa) data from mesh 
    //assumes flow factor file is called "flwa" and its first row is the number of elements in the mesh
    FILE *flwafile = fopen(flwafilename,"r");
    have_flwa = false;
    if (flwafile != NULL) have_flwa = true;
    if (have_flwa) {
      fseek(flwafile, 0, SEEK_SET); 
      fscanf(flwafile, "%lf", &temp); 
      flwa = new double[NumEles]; 
      fgets(buffer, 100, flwafile); 
      for (int i=0; i<NumEles; i++){
        fgets(buffer, 100, flwafile); 
        sscanf(buffer, "%lf", &flwa[i]); 
        //*out << "i: " << i << ", flwa: " << flwa[i] << std::endl; 
       }
     }
    //read in temperature data from mesh 
    //assumes temperature file is called "temp" and its first row is the number of elements in the mesh
    FILE *tempfile = fopen(tempfilename,"r");
    have_temp = false;
    if (tempfile != NULL) have_temp = true;
    if (have_temp) {
      fseek(tempfile, 0, SEEK_SET); 
      fscanf(tempfile, "%lf", &temp); 
      temper = new double[NumEles]; 
      fgets(buffer, 100, tempfile); 
      for (int i=0; i<NumEles; i++){
        fgets(buffer, 100, tempfile); 
        sscanf(buffer, "%lf", &temper[i]); 
        //*out << "i: " << i << ", temp: " << temper[i] << std::endl; 
       }
     }
    //read in basal friction (beta) data from mesh 
    //assumes basal friction file is called "beta" and its first row is the number of nodes  
    FILE *betafile = fopen(betafilename,"r");
    have_beta = false;
    if (betafile != NULL) have_beta = true;
    if (have_beta) {
      fseek(betafile, 0, SEEK_SET); 
      fscanf(betafile, "%lf", &temp); 
      beta = new double[NumNodes]; 
      fgets(buffer, 100, betafile); 
      for (int i=0; i<NumNodes; i++){
        fgets(buffer, 100, betafile); 
        sscanf(buffer, "%lf", &beta[i]); 
        //*out << "i: " << i << ", beta: " << beta[i] << std::endl; 
       }
     }
 
  elem_map = Teuchos::rcp(new Epetra_Map(-1, NumEles, globalElesID, 0, *comm)); //Distribute the elements according to the global element IDs
  node_map = Teuchos::rcp(new Epetra_Map(-1, NumNodes, globalNodesID, 0, *comm)); //Distribute the nodes according to the global node IDs 
  basal_face_map = Teuchos::rcp(new Epetra_Map(-1, NumBasalFaces, basalFacesID, 0, *comm)); //Distribute the elements according to the basal face IDs

  
  params->validateParameters(*getValidDiscretizationParameters(),0);


  std::string ebn="Element Block 0";
  partVec[0] = & metaData->declare_part(ebn, metaData->element_rank() );
  ebNameToIndex[ebn] = 0;

#ifdef ALBANY_SEACAS
  stk_classic::io::put_io_part_attribute(*partVec[0]);
#endif


  std::vector<std::string> nsNames;
  std::string nsn="Bottom";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
    stk_classic::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn="NodeSet0";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
    stk_classic::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn="NodeSet1";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
    stk_classic::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn="NodeSet2";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
    stk_classic::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn="NodeSet3";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
    stk_classic::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn="NodeSet4";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
    stk_classic::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn="NodeSet5";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
    stk_classic::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif


  std::vector<std::string> ssNames;
  std::string ssn="Basal";
  ssNames.push_back(ssn);
    ssPartVec[ssn] = & metaData->declare_part(ssn, metaData->side_rank() );
#ifdef ALBANY_SEACAS
    stk_classic::io::put_io_part_attribute(*ssPartVec[ssn]);
#endif

  stk_classic::mesh::fem::set_cell_topology<shards::Hexahedron<8> >(*partVec[0]);
  stk_classic::mesh::fem::set_cell_topology<shards::Quadrilateral<4> >(*ssPartVec[ssn]);

  numDim = 3;
  int cub = params->get("Cubature Degree",3);
  int worksetSizeMax = params->get("Workset Size",50);
  int worksetSize = this->computeWorksetSize(worksetSizeMax, elem_map->NumMyElements());

  const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();

  this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub,
                             nsNames, ssNames, worksetSize, partVec[0]->name(),
                             ebNameToIndex, this->interleavedOrdering));


}

Albany::AsciiSTKMeshStruct::~AsciiSTKMeshStruct()
{
  delete [] xyz; 
  if (have_sh) delete [] sh; 
  if (have_bf) delete [] bf; 
  delete [] eles; 
  delete [] globalElesID; 
  delete [] globalNodesID;
  delete [] basalFacesID; 
}

void
Albany::AsciiSTKMeshStruct::setFieldAndBulkData(
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

  stk_classic::mesh::PartVector nodePartVec;
  stk_classic::mesh::PartVector singlePartVec(1);
  stk_classic::mesh::PartVector emptyPartVec;
#ifdef OUTPUT_TO_SCREEN
  *out << "elem_map # elements: " << elem_map->NumMyElements() << std::endl; 
  *out << "node_map # elements: " << node_map->NumMyElements() << std::endl;
#endif 
  unsigned int ebNo = 0; //element block #??? 
  int sideID = 0;

  AbstractSTKFieldContainer::VectorFieldType* coordinates_field = fieldContainer->getCoordinatesField();
  AbstractSTKFieldContainer::ScalarFieldType* surfaceHeight_field = fieldContainer->getSurfaceHeightField();
  AbstractSTKFieldContainer::ScalarFieldType* flowFactor_field = fieldContainer->getFlowFactorField();
  AbstractSTKFieldContainer::ScalarFieldType* temperature_field = fieldContainer->getTemperatureField();
  AbstractSTKFieldContainer::ScalarFieldType* basal_friction_field = fieldContainer->getBasalFrictionField();

  if(!surfaceHeight_field) 
     have_sh = false;
  if(!flowFactor_field) 
     have_flwa = false;
  if(!temperature_field) 
     have_temp = false;
  if(!basal_friction_field) 
     have_beta = false;

  for (int i=0; i<elem_map->NumMyElements(); i++) {
     const unsigned int elem_GID = elem_map->GID(i);
     //std::cout << "elem_GID: " << elem_GID << std::endl; 
     stk_classic::mesh::EntityId elem_id = (stk_classic::mesh::EntityId) elem_GID;
     singlePartVec[0] = partVec[ebNo];
     stk_classic::mesh::Entity& elem  = bulkData->declare_entity(metaData->element_rank(), 1+elem_id, singlePartVec);
     //I am assuming the ASCII mesh is 1-based not 0-based, so no need to add 1 for STK mesh 
     stk_classic::mesh::Entity& llnode = bulkData->declare_entity(metaData->node_rank(), eles[i][0], nodePartVec);
     stk_classic::mesh::Entity& lrnode = bulkData->declare_entity(metaData->node_rank(), eles[i][1], nodePartVec);
     stk_classic::mesh::Entity& urnode = bulkData->declare_entity(metaData->node_rank(), eles[i][2], nodePartVec);
     stk_classic::mesh::Entity& ulnode = bulkData->declare_entity(metaData->node_rank(), eles[i][3], nodePartVec);
     stk_classic::mesh::Entity& llnodeb = bulkData->declare_entity(metaData->node_rank(), eles[i][4], nodePartVec);
     stk_classic::mesh::Entity& lrnodeb = bulkData->declare_entity(metaData->node_rank(), eles[i][5], nodePartVec);
     stk_classic::mesh::Entity& urnodeb = bulkData->declare_entity(metaData->node_rank(), eles[i][6], nodePartVec);
     stk_classic::mesh::Entity& ulnodeb = bulkData->declare_entity(metaData->node_rank(), eles[i][7], nodePartVec);
     bulkData->declare_relation(elem, llnode, 0);
     bulkData->declare_relation(elem, lrnode, 1);
     bulkData->declare_relation(elem, urnode, 2);
     bulkData->declare_relation(elem, ulnode, 3);
     bulkData->declare_relation(elem, llnodeb, 4);
     bulkData->declare_relation(elem, lrnodeb, 5);
     bulkData->declare_relation(elem, urnodeb, 6);
     bulkData->declare_relation(elem, ulnodeb, 7);
    

     double* coord;
     int node_GID;
     unsigned int node_LID;

     node_GID = eles[i][0]-1;
     node_LID = node_map->LID(node_GID);
     coord = stk_classic::mesh::field_data(*coordinates_field, llnode);
     coord[0] = xyz[node_LID][0];   coord[1] = xyz[node_LID][1];   coord[2] = xyz[node_LID][2];

     node_GID = eles[i][1]-1;
     node_LID = node_map->LID(node_GID);
     coord = stk_classic::mesh::field_data(*coordinates_field, lrnode);
     coord[0] = xyz[node_LID][0];   coord[1] = xyz[node_LID][1];   coord[2] = xyz[node_LID][2];

     node_GID = eles[i][2]-1;
     node_LID = node_map->LID(node_GID);
     coord = stk_classic::mesh::field_data(*coordinates_field, urnode);
     coord[0] = xyz[node_LID][0];   coord[1] = xyz[node_LID][1];   coord[2] = xyz[node_LID][2];

     node_GID = eles[i][3]-1;
     node_LID = node_map->LID(node_GID);
     coord = stk_classic::mesh::field_data(*coordinates_field, ulnode);
     coord[0] = xyz[node_LID][0];   coord[1] = xyz[node_LID][1];   coord[2] = xyz[node_LID][2];

     coord = stk_classic::mesh::field_data(*coordinates_field, llnodeb);
     node_GID = eles[i][4]-1;
     node_LID = node_map->LID(node_GID);
     coord[0] = xyz[node_LID][0];   coord[1] = xyz[node_LID][1];   coord[2] = xyz[node_LID][2];

     node_GID = eles[i][5]-1;
     node_LID = node_map->LID(node_GID);
     coord = stk_classic::mesh::field_data(*coordinates_field, lrnodeb);
     coord[0] = xyz[node_LID][0];   coord[1] = xyz[node_LID][1];   coord[2] = xyz[node_LID][2];

     coord = stk_classic::mesh::field_data(*coordinates_field, urnodeb);
     node_GID = eles[i][6]-1;
     node_LID = node_map->LID(node_GID);
     coord[0] = xyz[node_LID][0];   coord[1] = xyz[node_LID][1];   coord[2] = xyz[node_LID][2];

     coord = stk_classic::mesh::field_data(*coordinates_field, ulnodeb);
     node_GID = eles[i][7]-1;
     node_LID = node_map->LID(node_GID);
     coord[0] = xyz[node_LID][0];   coord[1] = xyz[node_LID][1];   coord[2] = xyz[node_LID][2];

#ifdef ALBANY_FELIX
     if (have_sh) {
       double* sHeight;
       sHeight = stk_classic::mesh::field_data(*surfaceHeight_field, llnode);
       node_GID = eles[i][0]-1;
       node_LID = node_map->LID(node_GID);
       sHeight[0] = sh[node_LID];

       sHeight = stk_classic::mesh::field_data(*surfaceHeight_field, lrnode);
       node_GID = eles[i][1]-1;
       node_LID = node_map->LID(node_GID);
       sHeight[0] = sh[node_LID];

       sHeight = stk_classic::mesh::field_data(*surfaceHeight_field, urnode);
       node_GID = eles[i][2]-1;
       node_LID = node_map->LID(node_GID);
       sHeight[0] = sh[node_LID];

       sHeight = stk_classic::mesh::field_data(*surfaceHeight_field, ulnode);
       node_GID = eles[i][3]-1;
       node_LID = node_map->LID(node_GID);
       sHeight[0] = sh[node_LID];

       sHeight = stk_classic::mesh::field_data(*surfaceHeight_field, llnodeb);
       node_GID = eles[i][4]-1;
       node_LID = node_map->LID(node_GID);
       sHeight[0] = sh[node_LID];

       sHeight = stk_classic::mesh::field_data(*surfaceHeight_field, lrnodeb);
       node_GID = eles[i][5]-1;
       node_LID = node_map->LID(node_GID);
       sHeight[0] = sh[node_LID];

       sHeight = stk_classic::mesh::field_data(*surfaceHeight_field, urnodeb);
       node_GID = eles[i][6]-1;
       node_LID = node_map->LID(node_GID);
       sHeight[0] = sh[node_LID];

       sHeight = stk_classic::mesh::field_data(*surfaceHeight_field, ulnodeb);
       node_GID = eles[i][7]-1;
       node_LID = node_map->LID(node_GID);
       sHeight[0] = sh[node_LID];
     }
     if (have_flwa) {
       double *flowFactor = stk_classic::mesh::field_data(*flowFactor_field, elem); 
       //i is elem_LID (element local ID);
       //*out << "i: " << i <<", flwa: " << flwa[i] << std::endl;  
       flowFactor[0] = flwa[i]; 
     }
     if (have_temp) {
       double *temperature = stk_classic::mesh::field_data(*temperature_field, elem); 
       //i is elem_LID (element local ID);
       //*out << "i: " << i <<", temp: " << temperature[i] << std::endl;  
       temperature[0] = temper[i]; 
     }
     if (have_beta) {
       double* bFriction; 
       bFriction = stk_classic::mesh::field_data(*basal_friction_field, llnode);
       node_GID = eles[i][0]-1;
       node_LID = node_map->LID(node_GID);
       bFriction[0] = beta[node_LID];

       bFriction = stk_classic::mesh::field_data(*basal_friction_field, lrnode);
       node_GID = eles[i][1]-1;
       node_LID = node_map->LID(node_GID);
       bFriction[0] = beta[node_LID];

       bFriction = stk_classic::mesh::field_data(*basal_friction_field, urnode);
       node_GID = eles[i][2]-1;
       node_LID = node_map->LID(node_GID);
       bFriction[0] = beta[node_LID];

       bFriction = stk_classic::mesh::field_data(*basal_friction_field, ulnode);
       node_GID = eles[i][3]-1;
       node_LID = node_map->LID(node_GID);
       bFriction[0] = beta[node_LID];

       bFriction = stk_classic::mesh::field_data(*basal_friction_field, llnodeb);
       node_GID = eles[i][4]-1;
       node_LID = node_map->LID(node_GID);
       bFriction[0] = beta[node_LID];

       bFriction = stk_classic::mesh::field_data(*basal_friction_field, lrnodeb);
       node_GID = eles[i][5]-1;
       node_LID = node_map->LID(node_GID);
       bFriction[0] = beta[node_LID];

       bFriction = stk_classic::mesh::field_data(*basal_friction_field, urnodeb);
       node_GID = eles[i][6]-1;
       node_LID = node_map->LID(node_GID);
       bFriction[0] = beta[node_LID];

       bFriction = stk_classic::mesh::field_data(*basal_friction_field, ulnodeb);
       node_GID = eles[i][7]-1;
       node_LID = node_map->LID(node_GID);
       bFriction[0] = beta[node_LID]; 
       }
#endif

     // If first node has z=0 and there is no basal face file provided, identify it as a Basal SS
     if (have_bf == false) {
       *out <<"No bf file specified...  setting basal boundary to z=0 plane..." << std::endl; 
       if ( xyz[eles[i][0]][2] == 0.0) {
          //std::cout << "sideID: " << sideID << std::endl; 
          singlePartVec[0] = ssPartVec["Basal"];
          stk_classic::mesh::EntityId side_id = (stk_classic::mesh::EntityId)(sideID);
          sideID++;

         stk_classic::mesh::Entity& side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
         bulkData->declare_relation(elem, side,  4 /*local side id*/);

         bulkData->declare_relation(side, llnode, 0);
         bulkData->declare_relation(side, ulnode, 3);
         bulkData->declare_relation(side, urnode, 2);
         bulkData->declare_relation(side, lrnode, 1);
       }
    }

    if (xyz[eles[i][0]-1][0] == 0.0) {
       singlePartVec[0] = nsPartVec["NodeSet0"];
       bulkData->change_entity_parts(llnode, singlePartVec); // node 0
       bulkData->change_entity_parts(ulnode, singlePartVec); // 3
       bulkData->change_entity_parts(llnodeb, singlePartVec); // 4
       bulkData->change_entity_parts(ulnodeb, singlePartVec); // 7
    }
    if (xyz[eles[i][1]-1][0] == 1.0) {
       singlePartVec[0] = nsPartVec["NodeSet1"];
       bulkData->change_entity_parts(lrnode, singlePartVec); // 1
       bulkData->change_entity_parts(urnode, singlePartVec); // 2
       bulkData->change_entity_parts(lrnodeb, singlePartVec); // 5
       bulkData->change_entity_parts(urnodeb, singlePartVec); // 6
    }
    if (xyz[eles[i][0]-1][1] == 0.0) {
       singlePartVec[0] = nsPartVec["NodeSet2"];
       bulkData->change_entity_parts(llnode, singlePartVec); // 0
       bulkData->change_entity_parts(lrnode, singlePartVec); // 1
       bulkData->change_entity_parts(llnodeb, singlePartVec); // 4
       bulkData->change_entity_parts(lrnodeb, singlePartVec); // 5
    }
    if (xyz[eles[i][2]-1][1] == 1.0) {
       singlePartVec[0] = nsPartVec["NodeSet3"];
       bulkData->change_entity_parts(ulnode, singlePartVec); // 3
       bulkData->change_entity_parts(urnode, singlePartVec); // 2
       bulkData->change_entity_parts(ulnodeb, singlePartVec); // 7
       bulkData->change_entity_parts(urnodeb, singlePartVec); // 6
    }
    if (xyz[eles[i][0]-1][2] == 0.0) {
       singlePartVec[0] = nsPartVec["NodeSet4"];
       bulkData->change_entity_parts(llnode, singlePartVec); // 0
       bulkData->change_entity_parts(lrnode, singlePartVec); // 1
       bulkData->change_entity_parts(ulnode, singlePartVec); // 3
       bulkData->change_entity_parts(urnode, singlePartVec); // 2
    }
    if (xyz[eles[i][4]-1][2] == 1.0) {
       singlePartVec[0] = nsPartVec["NodeSet5"];
       bulkData->change_entity_parts(llnodeb, singlePartVec); // 4
       bulkData->change_entity_parts(lrnodeb, singlePartVec); // 5
       bulkData->change_entity_parts(ulnodeb, singlePartVec); // 7
       bulkData->change_entity_parts(urnodeb, singlePartVec); // 6
    }

     // If first node has z=0, identify it as a Bottom NS
     if ( xyz[eles[i][0]-1][2] == 0.0) {
       singlePartVec[0] = nsPartVec["Bottom"];
       bulkData->change_entity_parts(llnode, singlePartVec); // 0
       bulkData->change_entity_parts(lrnode, singlePartVec); // 1
       bulkData->change_entity_parts(ulnode, singlePartVec); // 3
       bulkData->change_entity_parts(urnode, singlePartVec); // 2
     }
  }
  if (have_bf == true) {
    *out << "Setting basal surface connectivity from bf file provided..." << std::endl;  
    for (int i=0; i<basal_face_map->NumMyElements(); i++) {
       singlePartVec[0] = ssPartVec["Basal"];
       sideID = basal_face_map->GID(i); 
       stk_classic::mesh::EntityId side_id = (stk_classic::mesh::EntityId)(sideID);
       stk_classic::mesh::Entity& side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);

       const unsigned int elem_GID = bf[i][0];
       //std::cout << "elem_GID: " << elem_GID << std::endl; 
       stk_classic::mesh::EntityId elem_id = (stk_classic::mesh::EntityId) elem_GID;
       stk_classic::mesh::Entity& elem  = bulkData->declare_entity(metaData->element_rank(), elem_id, emptyPartVec);
       bulkData->declare_relation(elem, side,  4 /*local side id*/);
       stk_classic::mesh::Entity& llnode = bulkData->declare_entity(metaData->node_rank(), bf[i][1], nodePartVec);
       stk_classic::mesh::Entity& lrnode = bulkData->declare_entity(metaData->node_rank(), bf[i][2], nodePartVec);
       stk_classic::mesh::Entity& urnode = bulkData->declare_entity(metaData->node_rank(), bf[i][3], nodePartVec);
       stk_classic::mesh::Entity& ulnode = bulkData->declare_entity(metaData->node_rank(), bf[i][4], nodePartVec);
       
       bulkData->declare_relation(side, llnode, 0);
       bulkData->declare_relation(side, ulnode, 3);
       bulkData->declare_relation(side, urnode, 2);
       bulkData->declare_relation(side, lrnode, 1);
    }
  }

  bulkData->modification_end();
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::AsciiSTKMeshStruct::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getValidGenericSTKParameters("Valid ASCII_DiscParams");

  return validPL;
}
