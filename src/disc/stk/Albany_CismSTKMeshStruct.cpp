//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>

#include "Albany_CismSTKMeshStruct.hpp"
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


//Constructor for arrays passed from CISM through Albany-CISM interface
Albany::CismSTKMeshStruct::CismSTKMeshStruct(
                  const Teuchos::RCP<Teuchos::ParameterList>& params, 
                  const Teuchos::RCP<const Epetra_Comm>& comm, 
                  const double * xyz_at_nodes_Ptr, 
                  const int * global_node_id_owned_map_Ptr, 
                  const int * global_element_id_active_owned_map_Ptr, 
                  const int * global_element_conn_active_Ptr, 
                  const int *global_basal_face_active_owned_map_Ptr, 
                  const int * global_basal_face_conn_active_Ptr, 
                  const double * beta_at_nodes_Ptr, 
                  const double * surf_height_at_nodes_Ptr, 
                  const double * flwa_at_active_elements_Ptr,
                  const int nNodes, const int nElementsActive, 
                  const int nCellsActive, const int verbosity) : 
  GenericSTKMeshStruct(params,Teuchos::null,3),
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  hasRestartSol(false), 
  restartTime(0.0), 
  periodic(false)
{
  if (verbosity == 1 & comm->MyPID() == 0) std::cout <<"In Albany::CismSTKMeshStruct - double * array inputs!" << std::endl; 
  NumNodes = nNodes;  
  NumEles = nElementsActive; 
  NumBasalFaces = nCellsActive;
  debug_output_verbosity = verbosity;
  if (verbosity == 2) 
    std::cout <<"NumNodes = " << NumNodes << ", NumEles = "<< NumEles << ", NumBasalFaces = " << NumBasalFaces << std::endl; 
  xyz = new double[NumNodes][3]; 
  eles = new int[NumEles][8]; 
  bf = new int[NumBasalFaces][5]; //1st column of bf: element # that face belongs to, 2rd-5th columns of bf: connectivity (hard-coded for quad faces) 
  sh = new double[NumNodes]; 
  globalNodesID = new int[NumNodes];
  globalElesID = new int[NumEles];
  basalFacesID = new int[NumBasalFaces];
  flwa = new double[NumEles]; 
  beta = new double[NumNodes]; 
  //TO DO? pass in temper?  for now, flwa is passed instead of temper
  //temper = new double[NumEles]; 
  
  //check if optional input fields exist
  if (surf_height_at_nodes_Ptr != NULL) have_sh = true;
  else have_sh = false;  
  if (global_basal_face_active_owned_map_Ptr != NULL) have_bf = true; 
  else have_bf = false; 
  if (flwa_at_active_elements_Ptr != NULL) have_flwa = true; 
  else have_flwa = false; 
  if (beta_at_nodes_Ptr != NULL) have_beta = true; 
  else have_beta = false; 

  have_temp = false; //for now temperature field is not passed; flwa is passed instead

  for (int i=0; i<NumNodes; i++){
    globalNodesID[i] = global_node_id_owned_map_Ptr[i]-1;  
    for (int j=0; j<3; j++) 
      xyz[i][j] = xyz_at_nodes_Ptr[i + NumNodes*j]; 
    //*out << "i: " << i << ", x: " << xyz[i][0] << ", y: " << xyz[i][1] << ", z: " << xyz[i][2] << std::endl; 
  }
  if (have_sh) {
    for (int i=0; i<NumNodes; i++) 
      sh[i] = surf_height_at_nodes_Ptr[i]; 
  }
  if (have_beta) {
    for (int i=0; i<NumNodes; i++) {
      beta[i] = beta_at_nodes_Ptr[i]; 
      //*out << "beta[i] " << beta[i] << std::endl; 
    }
  }
 
  for (int i=0; i<NumEles; i++) { 
    globalElesID[i] = global_element_id_active_owned_map_Ptr[i]-1; 
    for (int j = 0; j<8; j++)
      eles[i][j] = global_element_conn_active_Ptr[i + nElementsActive*j];
    //*out << "elt # " << globalElesID[i] << ": " << eles[i][0] << " " << eles[i][1] << " " << eles[i][2] << " " << eles[i][3] << " " << eles[i][4] << " "
    //                      << eles[i][5] << " " << eles[i][6] << " " << eles[i][7] << std::endl; 
  }
  if (have_flwa) { 
    for (int i=0; i<NumEles; i++)  
      flwa[i] = flwa_at_active_elements_Ptr[i];
  }
  if (have_bf) { 
    for (int i=0; i<NumBasalFaces; i++) {
      basalFacesID[i] = global_basal_face_active_owned_map_Ptr[i]-1; 
      for (int j=0; j<5; j++) 
        bf[i][j] = global_basal_face_conn_active_Ptr[i + nCellsActive*j]; 
        //*out << "bf # " << basalFacesID[i] << ": " << bf[i][0] << " " << bf[i][1] << " " << bf[i][2] << " " << bf[i][3] << " " << bf[i][4] << std::endl; 
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


Albany::CismSTKMeshStruct::~CismSTKMeshStruct()
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
Albany::CismSTKMeshStruct::constructMesh(
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
  if (debug_output_verbosity == 2) {
    std::cout << "elem_map # elements: " << elem_map->NumMyElements() << std::endl; 
    std::cout << "node_map # elements: " << node_map->NumMyElements() << std::endl; 
  }
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
       if (debug_output_verbosity != 0) *out <<"No bf file specified...  setting basal boundary to z=0 plane..." << std::endl; 
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
  }

  if (have_bf == true) {
    if (debug_output_verbosity != 0) *out << "Setting basal surface connectivity from bf file provided..." << std::endl; 
    for (int i=0; i<basal_face_map->NumMyElements(); i++) {
       singlePartVec[0] = ssPartVec["Basal"];
       sideID = basal_face_map->GID(i); 
       stk_classic::mesh::EntityId side_id = (stk_classic::mesh::EntityId)(sideID);
       stk_classic::mesh::Entity& side  = bulkData->declare_entity(metaData->side_rank(),side_id+1, singlePartVec);
       const unsigned int elem_GID = bf[i][0];
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
Albany::CismSTKMeshStruct::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getValidGenericSTKParameters("Valid ASCII_DiscParams");

  return validPL;
}
