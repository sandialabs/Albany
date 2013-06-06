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

Albany::AsciiSTKMeshStruct::AsciiSTKMeshStruct(
                                             const Teuchos::RCP<Teuchos::ParameterList>& params, 
                                             const Teuchos::RCP<const Epetra_Comm>& comm) :
  GenericSTKMeshStruct(params,Teuchos::null,3),
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  periodic(false)
{
  if (comm->MyPID()==0){ // read in mesh data from ASCII file on processor 1 
    //read in coordinates of mesh -- right now hard coded for 3D
    //assumes mesh file is called "xyz" and its first row is the number of nodes  
    FILE *meshfile = fopen("xyz","r");
    double temp; 
    fseek(meshfile, 0, SEEK_SET); 
    fscanf(meshfile, "%lf", &temp); 
    NumNodes = int(temp); 
    cout << "numNodes: " << NumNodes << endl;  
    xyz = new double[NumNodes][3]; 
    char buffer[100];
    fgets(buffer, 100, meshfile); 
    for (int i=0; i<NumNodes; i++){
      fgets(buffer, 100, meshfile); 
      sscanf(buffer, "%lf %lf %lf", &xyz[i][0], &xyz[i][1], &xyz[i][2]); 
      *out << "i: " << i << ", x: " << xyz[i][0] << ", y: " << xyz[i][1] << ", z: " << xyz[i][2] << endl; 
     }
    //read in surface height data from mesh 
    //assumes mesh file is called "sh" and its first row is the number of nodes  
    FILE *shfile = fopen("sh","r");
    have_sh = false;
    if (shfile != NULL) have_sh = true;
    if (have_sh) {
      fseek(shfile, 0, SEEK_SET); 
      fscanf(shfile, "%lf", &temp); 
      int NumNodesSh = int(temp);
      cout << "NumNodesSh: " << NumNodesSh<< endl; 
      if (NumNodesSh != NumNodes) { 
           *out << "Error in AsciiSTKMeshStruct: sh file must have same number nodes as xyz file!  numNodes in xyz = " << NumNodes <<", numNodes in sh = "<< NumNodesSh  << endl;
          TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
            endl << "Error in AsciiSTKMeshStruct: sh file must have same number nodes as xyz file!  numNodes in xyz = " << NumNodes << ", numNodes in sh = "<< NumNodesSh << endl);
      }
      sh = new double[NumNodes]; 
      fgets(buffer, 100, shfile); 
      for (int i=0; i<NumNodes; i++){
        fgets(buffer, 100, shfile); 
        sscanf(buffer, "%lf", &sh[i]); 
        *out << "i: " << i << ", sh: " << sh[i] << endl; 
       }
     }
     //read in connectivity file -- right now hard coded for 3D hexes
     //assumes mesh file is called "eles" and its first row is the number of elements  
     FILE *confile = fopen("eles","r"); 
     fseek(confile, 0, SEEK_SET); 
     fscanf(confile, "%lf", &temp); 
     NumEles = int(temp); 
     cout << "numEles: " << NumEles << endl; 
     eles = new int[NumEles][8]; 
     fgets(buffer, 100, confile); 
     for (int i=0; i<NumEles; i++){
        fgets(buffer, 100, confile); 
        sscanf(buffer, "%i %i %i %i %i %i %i %i", &eles[i][0], &eles[i][1], &eles[i][2], &eles[i][3], &eles[i][4], &eles[i][5], &eles[i][6], &eles[i][7]);
        *out << "elt # " << i << ": " << eles[i][0] << " " << eles[i][1] << " " << eles[i][2] << " " << eles[i][3] << " " << eles[i][4] << " "
                          << eles[i][5] << " " << eles[i][6] << " " << eles[i][7] << endl; 
     }
    //read in basal face connectivity file from ascii file
    //assumes basal face connectivity file is called "bf" and its first row is the number of faces on basal boundary
    FILE *bffile = fopen("bf","r");
    have_bf = false;
    if (bffile != NULL) have_bf = true;
    if (have_bf) {
      fseek(bffile, 0, SEEK_SET); 
      fscanf(bffile, "%lf", &temp); 
      NumBasalFaces = int(temp); 
      cout << "numBasalFaces: " << NumBasalFaces << endl;  
      bf = new int[NumBasalFaces][5]; //1st column of bf: element # that face belongs to, 2rd-5th columns of bf: connectivity (hard-coded for quad faces) 
      fgets(buffer, 100, bffile); 
      for (int i=0; i<NumBasalFaces; i++){
        fgets(buffer, 100, bffile); 
        sscanf(buffer, "%i %i %i %i %i", &bf[i][0], &bf[i][1], &bf[i][2], &bf[i][3], &bf[i][4]); 
        *out << "face #:" << bf[i][0] << ", face conn:" << bf[i][1] << " " << bf[i][2] << " " << bf[i][3] << " " << bf[i][4] << endl; 
       }
     }
   }
 
  elem_map = Teuchos::rcp(new Epetra_Map(NumEles, 0, *comm)); // Distribute the elems equally
  basal_face_map = Teuchos::rcp(new Epetra_Map(NumBasalFaces, 0, *comm)); // Distribute the basal faces equally
  
  params->validateParameters(*getValidDiscretizationParameters(),0);


  std::string ebn="Element Block 0";
  partVec[0] = & metaData->declare_part(ebn, metaData->element_rank() );
  ebNameToIndex[ebn] = 0;

#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*partVec[0]);
#endif


  std::vector<std::string> nsNames;
  std::string nsn="Bottom";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn="NodeSet0";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn="NodeSet1";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn="NodeSet2";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn="NodeSet3";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn="NodeSet4";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  nsn="NodeSet5";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif


  std::vector<std::string> ssNames;
  std::string ssn="Basal";
  ssNames.push_back(ssn);
    ssPartVec[ssn] = & metaData->declare_part(ssn, metaData->side_rank() );
#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*ssPartVec[ssn]);
#endif

  stk::mesh::fem::set_cell_topology<shards::Hexahedron<8> >(*partVec[0]);
  stk::mesh::fem::set_cell_topology<shards::Quadrilateral<4> >(*ssPartVec[ssn]);

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

  stk::mesh::PartVector nodePartVec;
  stk::mesh::PartVector singlePartVec(1);
  stk::mesh::PartVector emptyPartVec;
  cout << "elem_map # elments: " << elem_map->NumMyElements() << endl; 
  unsigned int ebNo = 0; //element block #??? 
  int sideID = 0;

//  AbstractSTKFieldContainer::IntScalarFieldType* proc_rank_field = fieldContainer->getProcRankField();
  AbstractSTKFieldContainer::VectorFieldType* coordinates_field = fieldContainer->getCoordinatesField();
  AbstractSTKFieldContainer::ScalarFieldType* surfaceHeight_field = fieldContainer->getSurfaceHeightField();

  if(!surfaceHeight_field) 
     have_sh = false;

  for (int i=0; i<elem_map->NumMyElements(); i++) {
     const unsigned int elem_GID = elem_map->GID(i);
     cout << "elem_GID: " << elem_GID << endl; 
     stk::mesh::EntityId elem_id = (stk::mesh::EntityId) elem_GID;
     singlePartVec[0] = partVec[ebNo];
     stk::mesh::Entity& elem  = bulkData->declare_entity(metaData->element_rank(), 1+elem_id, singlePartVec);
     //I am assuming the ASCII mesh is 1-based not 0-based, so no need to add 1 for STK mesh 
     stk::mesh::Entity& llnode = bulkData->declare_entity(metaData->node_rank(), eles[elem_GID][0], nodePartVec);
     stk::mesh::Entity& lrnode = bulkData->declare_entity(metaData->node_rank(), eles[elem_GID][1], nodePartVec);
     stk::mesh::Entity& urnode = bulkData->declare_entity(metaData->node_rank(), eles[elem_GID][2], nodePartVec);
     stk::mesh::Entity& ulnode = bulkData->declare_entity(metaData->node_rank(), eles[elem_GID][3], nodePartVec);
     stk::mesh::Entity& llnodeb = bulkData->declare_entity(metaData->node_rank(), eles[elem_GID][4], nodePartVec);
     stk::mesh::Entity& lrnodeb = bulkData->declare_entity(metaData->node_rank(), eles[elem_GID][5], nodePartVec);
     stk::mesh::Entity& urnodeb = bulkData->declare_entity(metaData->node_rank(), eles[elem_GID][6], nodePartVec);
     stk::mesh::Entity& ulnodeb = bulkData->declare_entity(metaData->node_rank(), eles[elem_GID][7], nodePartVec);
     bulkData->declare_relation(elem, llnode, 0);
     bulkData->declare_relation(elem, lrnode, 1);
     bulkData->declare_relation(elem, urnode, 2);
     bulkData->declare_relation(elem, ulnode, 3);
     bulkData->declare_relation(elem, llnodeb, 4);
     bulkData->declare_relation(elem, lrnodeb, 5);
     bulkData->declare_relation(elem, urnodeb, 6);
     bulkData->declare_relation(elem, ulnodeb, 7);
    
/*
     if(proc_rank_field){
       int* p_rank = stk::mesh::field_data(*proc_rank_field, elem);
       p_rank[0] = comm->MyPID();
     }
*/

     double* coord;
     coord = stk::mesh::field_data(*coordinates_field, llnode);
     coord[0] = xyz[eles[elem_GID][0]-1][0];   coord[1] = xyz[eles[elem_GID][0]-1][1];   coord[2] = xyz[eles[elem_GID][0]-1][2]; 
     coord = stk::mesh::field_data(*coordinates_field, lrnode);
     coord[0] = xyz[eles[elem_GID][1]-1][0];   coord[1] = xyz[eles[elem_GID][1]-1][1];   coord[2] = xyz[eles[elem_GID][1]-1][2]; 
     coord = stk::mesh::field_data(*coordinates_field, urnode);
     coord[0] = xyz[eles[elem_GID][2]-1][0];   coord[1] = xyz[eles[elem_GID][2]-1][1];   coord[2] = xyz[eles[elem_GID][2]-1][2]; 
     coord = stk::mesh::field_data(*coordinates_field, ulnode);
     coord[0] = xyz[eles[elem_GID][3]-1][0];   coord[1] = xyz[eles[elem_GID][3]-1][1];   coord[2] = xyz[eles[elem_GID][3]-1][2]; 

     coord = stk::mesh::field_data(*coordinates_field, llnodeb);
     coord[0] = xyz[eles[elem_GID][4]-1][0];   coord[1] = xyz[eles[elem_GID][4]-1][1];   coord[2] = xyz[eles[elem_GID][4]-1][2]; 
     coord = stk::mesh::field_data(*coordinates_field, lrnodeb);
     coord[0] = xyz[eles[elem_GID][5]-1][0];   coord[1] = xyz[eles[elem_GID][5]-1][1];   coord[2] = xyz[eles[elem_GID][5]-1][2]; 
     coord = stk::mesh::field_data(*coordinates_field, urnodeb);
     coord[0] = xyz[eles[elem_GID][6]-1][0];   coord[1] = xyz[eles[elem_GID][6]-1][1];   coord[2] = xyz[eles[elem_GID][6]-1][2]; 
     coord = stk::mesh::field_data(*coordinates_field, ulnodeb);
     coord[0] = xyz[eles[elem_GID][7]-1][0];   coord[1] = xyz[eles[elem_GID][7]-1][1];   coord[2] = xyz[eles[elem_GID][7]-1][2]; 

#ifdef ALBANY_FELIX
     if (have_sh) {
     double* sHeight;
     sHeight = stk::mesh::field_data(*surfaceHeight_field, llnode);
     sHeight[0] = sh[eles[elem_GID][0]-1];
     sHeight = stk::mesh::field_data(*surfaceHeight_field, lrnode);
     sHeight[0] = sh[eles[elem_GID][1]-1];
     sHeight = stk::mesh::field_data(*surfaceHeight_field, urnode);
     sHeight[0] = sh[eles[elem_GID][2]-1];
     sHeight = stk::mesh::field_data(*surfaceHeight_field, ulnode);
     sHeight[0] = sh[eles[elem_GID][3]-1];

     sHeight = stk::mesh::field_data(*surfaceHeight_field, llnodeb);
     sHeight[0] = sh[eles[elem_GID][4]-1];
     sHeight = stk::mesh::field_data(*surfaceHeight_field, lrnodeb);
     sHeight[0] = sh[eles[elem_GID][5]-1];
     sHeight = stk::mesh::field_data(*surfaceHeight_field, urnodeb);
     sHeight[0] = sh[eles[elem_GID][6]-1];
     sHeight = stk::mesh::field_data(*surfaceHeight_field, ulnodeb);
     sHeight[0] = sh[eles[elem_GID][7]-1];
     }
#endif

     // If first node has z=0 and there is no basal face file provided, identify it as a Basal SS
     if (have_bf == false) {
       if ( xyz[eles[elem_GID][0]][2] == 0.0) {
          cout << "sideID: " << sideID << endl; 
          singlePartVec[0] = ssPartVec["Basal"];
          stk::mesh::EntityId side_id = (stk::mesh::EntityId)(sideID);
          sideID++;

         stk::mesh::Entity& side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);
         bulkData->declare_relation(elem, side,  4 /*local side id*/);

         bulkData->declare_relation(side, llnode, 0);
         bulkData->declare_relation(side, ulnode, 3);
         bulkData->declare_relation(side, urnode, 2);
         bulkData->declare_relation(side, lrnode, 1);
       }
    }

    if (xyz[eles[elem_GID][0]-1][0] == 0.0) {
       singlePartVec[0] = nsPartVec["NodeSet0"];
       bulkData->change_entity_parts(llnode, singlePartVec); // node 0
       bulkData->change_entity_parts(ulnode, singlePartVec); // 3
       bulkData->change_entity_parts(llnodeb, singlePartVec); // 4
       bulkData->change_entity_parts(ulnodeb, singlePartVec); // 7
    }
    if (xyz[eles[elem_GID][1]-1][0] == 1.0) {
       singlePartVec[0] = nsPartVec["NodeSet1"];
       bulkData->change_entity_parts(lrnode, singlePartVec); // 1
       bulkData->change_entity_parts(urnode, singlePartVec); // 2
       bulkData->change_entity_parts(lrnodeb, singlePartVec); // 5
       bulkData->change_entity_parts(urnodeb, singlePartVec); // 6
    }
    if (xyz[eles[elem_GID][0]-1][1] == 0.0) {
       singlePartVec[0] = nsPartVec["NodeSet2"];
       bulkData->change_entity_parts(llnode, singlePartVec); // 0
       bulkData->change_entity_parts(lrnode, singlePartVec); // 1
       bulkData->change_entity_parts(llnodeb, singlePartVec); // 4
       bulkData->change_entity_parts(lrnodeb, singlePartVec); // 5
    }
    if (xyz[eles[elem_GID][2]-1][1] == 1.0) {
       singlePartVec[0] = nsPartVec["NodeSet3"];
       bulkData->change_entity_parts(ulnode, singlePartVec); // 3
       bulkData->change_entity_parts(urnode, singlePartVec); // 2
       bulkData->change_entity_parts(ulnodeb, singlePartVec); // 7
       bulkData->change_entity_parts(urnodeb, singlePartVec); // 6
    }
    if (xyz[eles[elem_GID][0]-1][2] == 0.0) {
       singlePartVec[0] = nsPartVec["NodeSet4"];
       bulkData->change_entity_parts(llnode, singlePartVec); // 0
       bulkData->change_entity_parts(lrnode, singlePartVec); // 1
       bulkData->change_entity_parts(ulnode, singlePartVec); // 3
       bulkData->change_entity_parts(urnode, singlePartVec); // 2
    }
    if (xyz[eles[elem_GID][4]-1][2] == 1.0) {
       singlePartVec[0] = nsPartVec["NodeSet5"];
       bulkData->change_entity_parts(llnodeb, singlePartVec); // 4
       bulkData->change_entity_parts(lrnodeb, singlePartVec); // 5
       bulkData->change_entity_parts(ulnodeb, singlePartVec); // 7
       bulkData->change_entity_parts(urnodeb, singlePartVec); // 6
    }

     // If first node has z=0, identify it as a Bottom NS
     if ( xyz[eles[elem_GID][0]-1][2] == 0.0) {
       singlePartVec[0] = nsPartVec["Bottom"];
       bulkData->change_entity_parts(llnode, singlePartVec); // 0
       bulkData->change_entity_parts(lrnode, singlePartVec); // 1
       bulkData->change_entity_parts(ulnode, singlePartVec); // 3
       bulkData->change_entity_parts(urnode, singlePartVec); // 2
     }
  }
  if (have_bf == true) {
    *out << "Setting basal surface connectivity from bf file provided..." << endl;  
    for (int i=0; i<basal_face_map->NumMyElements(); i++) {
       singlePartVec[0] = ssPartVec["Basal"];
       sideID = basal_face_map->GID(i); 
       stk::mesh::EntityId side_id = (stk::mesh::EntityId)(sideID);
       stk::mesh::Entity& side  = bulkData->declare_entity(metaData->side_rank(), 1 + side_id, singlePartVec);

       const unsigned int elem_GID = bf[i][0];
       //cout << "elem_GID: " << elem_GID << endl; 
       stk::mesh::EntityId elem_id = (stk::mesh::EntityId) elem_GID;
       stk::mesh::Entity& elem  = bulkData->declare_entity(metaData->element_rank(), elem_id, emptyPartVec);
       bulkData->declare_relation(elem, side,  4 /*local side id*/);
       stk::mesh::Entity& llnode = bulkData->declare_entity(metaData->node_rank(), bf[i][1], nodePartVec);
       stk::mesh::Entity& lrnode = bulkData->declare_entity(metaData->node_rank(), bf[i][2], nodePartVec);
       stk::mesh::Entity& urnode = bulkData->declare_entity(metaData->node_rank(), bf[i][3], nodePartVec);
       stk::mesh::Entity& ulnode = bulkData->declare_entity(metaData->node_rank(), bf[i][4], nodePartVec);
       
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
