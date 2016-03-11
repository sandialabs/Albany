//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
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
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Selector.hpp>

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

#include <Albany_STKNodeSharing.hpp>

//#include <stk_mesh/fem/FEMHelpers.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "Albany_Utils.hpp"


//FIXME: replace double * with Teuchos::Arrays. 

//Constructor for arrays passed from CISM through Albany-CISM interface
Albany::CismSTKMeshStruct::CismSTKMeshStruct(
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
                  const double * xyz_at_nodes_Ptr,
                  const int * global_node_id_owned_map_Ptr,
                  const int * global_element_id_active_owned_map_Ptr,
                  const int * global_element_conn_active_Ptr,
                  const int * global_basal_face_active_owned_map_Ptr,
                  const int * global_top_face_active_owned_map_Ptr,
                  const int * global_basal_face_conn_active_Ptr,
                  const int * global_top_face_conn_active_Ptr,
                  const int * global_west_face_active_owned_map_Ptr,
                  const int * global_west_face_conn_active_Ptr,
                  const int * global_east_face_active_owned_map_Ptr,
                  const int * global_east_face_conn_active_Ptr,
                  const int * global_south_face_active_owned_map_Ptr,
                  const int * global_south_face_conn_active_Ptr,
                  const int * global_north_face_active_owned_map_Ptr,
                  const int * global_north_face_conn_active_Ptr,
                  const int * dirichlet_node_mask_Ptr,
                  const double * uvel_at_nodes_Ptr,
                  const double * vvel_at_nodes_Ptr, 
                  const double * beta_at_nodes_Ptr,
                  const double * surf_height_at_nodes_Ptr,
                  const double * dsurf_height_at_nodes_dx_Ptr,
                  const double * dsurf_height_at_nodes_dy_Ptr,
                  const double * thick_at_nodes_Ptr,
                  const double * flwa_at_active_elements_Ptr,
                  const int nNodes, const int nElementsActive,
                  const int nCellsActive, const int nWestFacesActive, 
                  const int nEastFacesActive, const int nSouthFacesActive, 
                  const int nNorthFacesActive, 
                  const int verbosity) :
  GenericSTKMeshStruct(params,Teuchos::null,3),
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  hasRestartSol(false),
  restartTime(0.0),
  periodic(false)
{
  if (verbosity == 1 & commT->getRank() == 0) std::cout <<"In Albany::CismSTKMeshStruct - double * array inputs!" << std::endl;
  NumNodes = nNodes;
  NumEles = nElementsActive;
  NumBasalFaces = nCellsActive;
  NumWestFaces =  nWestFacesActive; 
  NumEastFaces =  nEastFacesActive; 
  NumSouthFaces = nSouthFacesActive; 
  NumNorthFaces = nNorthFacesActive; 
  debug_output_verbosity = verbosity;
  if (verbosity == 2) {
    std::cout <<"Proc #" << commT->getRank() << ", NumNodes = " << NumNodes << ", NumEles = "<< NumEles << ", NumBasalFaces = " << NumBasalFaces 
               <<", NumWestFaces = " << NumWestFaces << ", NumEastFaces = "<< NumEastFaces 
              << ", NumSouthFaces = " << NumSouthFaces << ", NumNorthFaces = " << NumNorthFaces <<  std::endl;
  }
  xyz = new double[NumNodes][3];
  eles = new int[NumEles][8];
  dirichletNodeMask = new int[NumNodes];
  //1st column of bf: element # that face belongs to, 2rd-5th columns of bf: connectivity (hard-coded for quad faces)
  bf = new int[NumBasalFaces][5]; 
  tf = new int[NumBasalFaces][5]; 
  wf = new int[NumWestFaces][5]; 
  ef = new int[NumEastFaces][5]; 
  sf = new int[NumSouthFaces][5]; 
  nf = new int[NumNorthFaces][5]; 
  sh = new double[NumNodes];
  thck = new double[NumNodes];
  shGrad = new double[NumNodes][2];
  Teuchos::Array<GO> globalNodesID(NumNodes); // local; doesn't have to be class data unless desired
  Teuchos::Array<GO> globalElesID(NumEles);
  Teuchos::Array<GO> basalFacesID(NumBasalFaces); 
  Teuchos::Array<GO> topFacesID(NumBasalFaces);
  Teuchos::Array<GO> westFacesID(NumWestFaces); 
  Teuchos::Array<GO> eastFacesID(NumEastFaces); 
  Teuchos::Array<GO> southFacesID(NumSouthFaces); 
  Teuchos::Array<GO> northFacesID(NumNorthFaces); 
  flwa = new double[NumEles];
  beta = new double[NumNodes];
  uvel = new double[NumNodes];
  vvel = new double[NumNodes];

  //check if optional input fields exist
  if (surf_height_at_nodes_Ptr != NULL) have_sh = true;
  else have_sh = false;
  if (thick_at_nodes_Ptr != NULL) have_thck = true;
  else have_thck = false;
  if (dsurf_height_at_nodes_dx_Ptr != NULL && dsurf_height_at_nodes_dy_Ptr != NULL) have_shGrad = true;
  else have_shGrad = false;
  if (global_basal_face_active_owned_map_Ptr != NULL) have_bf = true; 
  else  have_bf = false;
  if (global_top_face_active_owned_map_Ptr != NULL) have_tf = true; 
  else  have_tf = false;
  if (flwa_at_active_elements_Ptr != NULL) have_flwa = true;
  else have_flwa = false;
  if (beta_at_nodes_Ptr != NULL) have_beta = true;
  else have_beta = false;
  if (global_west_face_active_owned_map_Ptr != NULL && NumWestFaces > 0) have_wf = true; 
  else have_wf = false; 
  if (global_east_face_active_owned_map_Ptr != NULL && NumEastFaces > 0) have_ef = true;
  else have_ef = false;
  if (global_south_face_active_owned_map_Ptr != NULL && NumSouthFaces > 0) have_sf = true;
  else have_sf = false;
  if (global_north_face_active_owned_map_Ptr != NULL && NumNorthFaces > 0) have_nf = true;
  else have_nf = false;
  if (dirichlet_node_mask_Ptr != NULL) have_dirichlet = true; 
  else have_dirichlet = false; 

  for (int i=0; i<NumNodes; i++){
    globalNodesID[i] = global_node_id_owned_map_Ptr[i]-1;
    for (int j=0; j<3; j++)
      xyz[i][j] = xyz_at_nodes_Ptr[i + NumNodes*j];
    //*out << "i: " << i << ", x: " << xyz[i][0] 
         //<< ", y: " << xyz[i][1] << ", z: " << xyz[i][2] << std::endl;
  }
  if (have_sh) {
    for (int i=0; i<NumNodes; i++)
      sh[i] = surf_height_at_nodes_Ptr[i];
  }
  if (have_thck) {
    for (int i=0; i<NumNodes; i++)
      thck[i] = thick_at_nodes_Ptr[i];
  }
  if (have_shGrad) {
    for (int i=0; i<NumNodes; i++){
      shGrad[i][0] = dsurf_height_at_nodes_dx_Ptr[i];
      shGrad[i][1] = dsurf_height_at_nodes_dy_Ptr[i];
    }
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
    //*out << "elt # " << globalElesID[i] << ": " << eles[i][0] 
    //     << " " << eles[i][1] << " " << eles[i][2] << " " 
    //     << eles[i][3] << " " << eles[i][4] << " "
    //     << eles[i][5] << " " << eles[i][6] << " " << eles[i][7] << std::endl;
  }
  if (have_dirichlet) {
    for (int i=0; i<NumNodes; i++) {
      dirichletNodeMask[i] = dirichlet_node_mask_Ptr[i];
      uvel[i] = uvel_at_nodes_Ptr[i]; 
      vvel[i] = vvel_at_nodes_Ptr[i]; 
      //*out << "i: " << i << ", x: " << xyz[i][0] 
      //     << ", y: " << xyz[i][1] << ", z: " << xyz[i][2] << ", dirichlet: " << dirichletNodeMask[i] << std::endl;
      /*if (abs(uvel[i]) > 0) {
      *out << "i: " << i << ", x: " << xyz[i][0] 
           << ", y: " << xyz[i][1] << ", z: " << xyz[i][2] << ", uvel: " << uvel[i] << ", vvel: " << vvel[i] << std::endl;
      }*/
    }
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
  if (have_tf) {
    for (int i=0; i<NumBasalFaces; i++) {
      topFacesID[i] = global_top_face_active_owned_map_Ptr[i]-1;
      for (int j=0; j<5; j++)
        tf[i][j] = global_top_face_conn_active_Ptr[i + nCellsActive*j];
    }
  }
  if (have_wf) {
    for (int i=0; i<NumWestFaces; i++) {
       westFacesID[i] = global_west_face_active_owned_map_Ptr[i]-1;    
       for (int j=0; j<5; j++) 
         wf[i][j] = global_west_face_conn_active_Ptr[i + NumWestFaces*j]; 
    }
  }
  if (have_ef) {
    for (int i=0; i<NumEastFaces; i++) {
       eastFacesID[i] = global_east_face_active_owned_map_Ptr[i]-1;    
       for (int j=0; j<5; j++) 
         ef[i][j] = global_east_face_conn_active_Ptr[i + NumEastFaces*j]; 
    }
  }
  if (have_sf) {
    for (int i=0; i<NumSouthFaces; i++) {
       southFacesID[i] = global_south_face_active_owned_map_Ptr[i]-1;    
       for (int j=0; j<5; j++)  
         sf[i][j] = global_south_face_conn_active_Ptr[i + NumSouthFaces*j];
        /*if (commT->getRank() == 0) { 
          *out << "proc 0, sf # " << southFacesID[i] << ": " << sf[i][0] << " " << sf[i][1] << " " << sf[i][2] << " " << sf[i][3] << " " << sf[i][4] << std::endl; }
        if (commT->getRank() == 1) { 
          *out << "proc 1, sf # " << southFacesID[i] << ": " << sf[i][0] << " " << sf[i][1] << " " << sf[i][2] << " " << sf[i][3] << " " << bf[i][4] << std::endl; }
        if (commT->getRank() == 2) { 
          *out << "proc 2, sf # " << southFacesID[i] << ": " << sf[i][0] << " " << sf[i][1] << " " << sf[i][2] << " " << sf[i][3] << " " << sf[i][4] << std::endl; }
        if (commT->getRank() == 3) { 
          *out << "proc 3, sf # " << southFacesID[i] << ": " << sf[i][0] << " " << sf[i][1] << " " << sf[i][2] << " " << sf[i][3] << " " << sf[i][4] << std::endl; } */
    }
  }
  if (have_nf) {
    for (int i=0; i<NumNorthFaces; i++) {
       northFacesID[i] = global_north_face_active_owned_map_Ptr[i]-1;    
       for (int j=0; j<5; j++) 
         nf[i][j] = global_north_face_conn_active_Ptr[i + NumNorthFaces*j]; 
    }
  }


  //Distribute the elements according to the global element IDs
  elem_mapT = Teuchos::rcp(new Tpetra_Map(NumEles, globalElesID, 0, commT)); 
  //Distribute the nodes according to the global node IDs
  node_mapT = Teuchos::rcp(new Tpetra_Map(NumNodes, globalNodesID, 0, commT));
  //Distribute the elements according to the basal face IDs
  basal_face_mapT = Teuchos::rcp(new Tpetra_Map(NumBasalFaces, basalFacesID, 0, commT));
  //Distribute the elements according to the top face IDs
  top_face_mapT = Teuchos::rcp(new Tpetra_Map(NumBasalFaces, topFacesID, 0, commT));
  //Distribute the elements according to the lateral face IDs
  west_face_mapT = Teuchos::rcp(new Tpetra_Map(NumWestFaces, westFacesID, 0, commT));
  east_face_mapT = Teuchos::rcp(new Tpetra_Map(NumEastFaces, eastFacesID, 0, commT));
  south_face_mapT = Teuchos::rcp(new Tpetra_Map(NumSouthFaces, southFacesID, 0, commT));
  north_face_mapT = Teuchos::rcp(new Tpetra_Map(NumNorthFaces, northFacesID, 0, commT));

  params->validateParameters(*getValidDiscretizationParameters(),0);


  std::string ebn="Element Block 0";
  partVec[0] = & metaData->declare_part(ebn, stk::topology::ELEMENT_RANK );
  ebNameToIndex[ebn] = 0;

#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*partVec[0]);
#endif


  std::vector<std::string> nsNames;
  std::string nsn="NodeSetDirichlet";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = & metaData->declare_part(nsn, stk::topology::NODE_RANK );
#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif


  std::vector<std::string> ssNames;
  std::string ssnBasal="Basal";
  std::string ssnTop="Top";
  std::string ssnLateral="Lateral";

  ssNames.push_back(ssnBasal);
  ssNames.push_back(ssnTop);
  ssNames.push_back(ssnLateral);

  ssPartVec[ssnBasal] = & metaData->declare_part(ssnBasal, metaData->side_rank() );
  ssPartVec[ssnTop] = & metaData->declare_part(ssnTop, metaData->side_rank() );
  ssPartVec[ssnLateral] = & metaData->declare_part(ssnLateral, metaData->side_rank() );
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*ssPartVec[ssnBasal]);
  stk::io::put_io_part_attribute(*ssPartVec[ssnTop]);
  stk::io::put_io_part_attribute(*ssPartVec[ssnLateral]);
#endif
  stk::mesh::set_cell_topology<shards::Hexahedron<8> >(*partVec[0]);
  stk::mesh::set_cell_topology<shards::Quadrilateral<4> >(*ssPartVec[ssnBasal]);
  stk::mesh::set_cell_topology<shards::Quadrilateral<4> >(*ssPartVec[ssnTop]);
  stk::mesh::set_cell_topology<shards::Quadrilateral<4> >(*ssPartVec[ssnLateral]);

  numDim = 3;
  int cub = params->get("Cubature Degree",3);
  int worksetSizeMax = params->get("Workset Size",50);
  int worksetSize = this->computeWorksetSize(worksetSizeMax, elem_mapT->getNodeNumElements());

  const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();

  this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub,
                             nsNames, ssNames, worksetSize, partVec[0]->name(),
                             ebNameToIndex, this->interleavedOrdering));

}


Albany::CismSTKMeshStruct::~CismSTKMeshStruct()
{
  delete [] xyz;
  delete [] dirichletNodeMask;
  delete [] sh;
  delete [] thck;
  delete [] shGrad;
  delete [] bf;
  delete [] tf;
  delete [] wf;
  delete [] ef;
  delete [] sf;
  delete [] nf;
  delete [] eles;
  delete [] uvel; 
  delete [] vvel; 
}

void
Albany::CismSTKMeshStruct::constructMesh(
                                               const Teuchos::RCP<const Teuchos_Comm>& commT,
                                               const Teuchos::RCP<Teuchos::ParameterList>& params,
                                               const unsigned int neq_,
                                               const AbstractFieldContainer::FieldContainerRequirements& req,
                                               const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                                               const unsigned int worksetSize)
{
  this->SetupFieldData(commT, neq_, req, sis, worksetSize);

  metaData->commit();

  bulkData->modification_begin(); // Begin modifying the mesh

  stk::mesh::PartVector nodePartVec;
  stk::mesh::PartVector singlePartVec(1);
  stk::mesh::PartVector emptyPartVec;
  if (debug_output_verbosity == 2) {
    std::cout << "elem_mapT # elements: " << elem_mapT->getNodeNumElements() << std::endl;
    std::cout << "node_mapT # elements: " << node_mapT->getNodeNumElements() << std::endl;
  }
  unsigned int ebNo = 0; //element block #???
  int sideID = 0;

  typedef AbstractSTKFieldContainer::ScalarFieldType ScalarFieldType;
  typedef AbstractSTKFieldContainer::VectorFieldType VectorFieldType;
  typedef AbstractSTKFieldContainer::QPScalarFieldType ElemScalarFieldType;

  VectorFieldType* coordinates_field = fieldContainer->getCoordinatesField();
  ScalarFieldType* surfaceHeight_field = metaData->get_field<ScalarFieldType>(stk::topology::NODE_RANK, "surface_height");
  ScalarFieldType* thickness_field = metaData->get_field<ScalarFieldType>(stk::topology::NODE_RANK, "thickness");
  ScalarFieldType* dsurfaceHeight_dx_field = metaData->get_field<ScalarFieldType>(stk::topology::NODE_RANK, "xgrad_surface_height");
  ScalarFieldType* dsurfaceHeight_dy_field = metaData->get_field<ScalarFieldType>(stk::topology::NODE_RANK, "ygrad_surface_height");
  ElemScalarFieldType* flowFactor_field = metaData->get_field<ElemScalarFieldType>(stk::topology::ELEMENT_RANK, "flow_factor");
  ElemScalarFieldType* temperature_field = metaData->get_field<ElemScalarFieldType>(stk::topology::ELEMENT_RANK, "temperature");
  ScalarFieldType* basal_friction_field = metaData->get_field<ScalarFieldType>(stk::topology::NODE_RANK, "basal_friction");
  VectorFieldType* dirichlet_field = metaData->get_field<VectorFieldType>(stk::topology::NODE_RANK, "dirichlet_field");

  if(!surfaceHeight_field)
     have_sh = false;
  if(!thickness_field)
     have_thck = false;
  if(!dsurfaceHeight_dx_field || !dsurfaceHeight_dy_field)
     have_shGrad = false;
  if(!flowFactor_field)
     have_flwa = false;
  if(!basal_friction_field)
     have_beta = false;

  double* coord;
  int node_GID;
  unsigned int node_LID;

  for (int i=0; i<elem_mapT->getNodeNumElements(); i++) {
     const unsigned int elem_GID = elem_mapT->getGlobalElement(i);
     stk::mesh::EntityId elem_id = (stk::mesh::EntityId) elem_GID;
     singlePartVec[0] = partVec[ebNo];
     //I am assuming the ASCII mesh is 1-based not 0-based, so no need to add 1 for STK mesh 
     stk::mesh::Entity elem  = bulkData->declare_entity(stk::topology::ELEMENT_RANK, 1+elem_id, singlePartVec);
     for (int j=0; j<8; j++) { //loop over 8 nodes of each element
     //Set element connectivity and coordinates
       stk::mesh::Entity node = bulkData->declare_entity(stk::topology::NODE_RANK, eles[i][j], nodePartVec);
       bulkData->declare_relation(elem, node, j);
       
       node_GID = eles[i][j]-1;
       node_LID = node_mapT->getLocalElement(node_GID);
       coord = stk::mesh::field_data(*coordinates_field, node);
       coord[0] = xyz[node_LID][0];   coord[1] = xyz[node_LID][1];   coord[2] = xyz[node_LID][2];
       //set surface height
       if (have_sh) {
         double* sHeight;
         sHeight = stk::mesh::field_data(*surfaceHeight_field, node);
         sHeight[0] = sh[node_LID];
       }
       //set thickness field
       if (have_thck) {
         double* thickness;
         thickness = stk::mesh::field_data(*thickness_field, node);
         thickness[0] = thck[node_LID];
       }
       //set gradients of surface height
       if (have_shGrad) {
         double* dsHeight_dx; 
         double* dsHeight_dy;
         dsHeight_dx = stk::mesh::field_data(*dsurfaceHeight_dx_field, node);
         dsHeight_dy = stk::mesh::field_data(*dsurfaceHeight_dy_field, node);
         dsHeight_dx[0] = shGrad[node_LID][0];
         dsHeight_dy[0] = shGrad[node_LID][1];
       }
       //set Dirichlet BCs to those passed from CISM.
       if (have_dirichlet) {
         double* dirichlet = stk::mesh::field_data(*dirichlet_field,node);
         dirichlet[0] = uvel[node_LID];
         dirichlet[1] = vvel[node_LID];
       }
       //set basal friction
       if (have_beta) { 
         double* bFriction; 
         bFriction = stk::mesh::field_data(*basal_friction_field, node);
         bFriction[0] = beta[node_LID];
      }
     }
     
     //Set Dirichlet nodesets 
     singlePartVec[0] = nsPartVec["NodeSetDirichlet"];
     for (int j=0; j<8; j++) { //loop over 8 nodes of each element
       node_GID = eles[i][j]-1;
       node_LID = node_mapT->getLocalElement(node_GID);
       if (dirichletNodeMask[node_LID] == 1) {
         stk::mesh::Entity node = bulkData->declare_entity(stk::topology::NODE_RANK, eles[i][j], nodePartVec);
         bulkData->change_entity_parts(node, singlePartVec); 
       }
     }

     //set fields that live at the elements
     if (have_flwa) {
       double *flowFactor = stk::mesh::field_data(*flowFactor_field, elem); 
       //i is elem_LID (element local ID);
       //*out << "i: " << i <<", flwa: " << flwa[i] << std::endl;
       flowFactor[0] = flwa[i];
       //Fill temperature field from flowRate
       //For CISM-Albany runs, flowRate will always be passed, not temperature.  
       double *temperature = stk::mesh::field_data(*temperature_field, elem);
       //This is the inverse of the temperature-flowRate relationship; see FELIX_ViscosityFO_Def.hpp .
       if (flwa[i] < 1.57349384110036e-05)
         temperature[0] = 6.0e4/log(1.3e7/flwa[i])/8.314;
       else 
         temperature[0] = 1.39e5/log(6.26e22/flwa[i])/8.314;
     }
     
  }

  //set basal face connectivity
  if (have_bf == true) {
    if (debug_output_verbosity != 0) *out << "Setting basal surface connectivity from data provided..." << std::endl;
    singlePartVec[0] = ssPartVec["Basal"];
    for (int i=0; i<basal_face_mapT->getNodeNumElements(); i++) {
       sideID = basal_face_mapT->getGlobalElement(i);
       stk::mesh::EntityId side_id = (stk::mesh::EntityId)(sideID);
       stk::mesh::Entity side  = bulkData->declare_entity(metaData->side_rank(),side_id+1, singlePartVec);
       const unsigned int elem_GID = bf[i][0];
       stk::mesh::EntityId elem_id = (stk::mesh::EntityId) elem_GID;
       stk::mesh::Entity elem  = bulkData->declare_entity(stk::topology::ELEMENT_RANK, elem_id, emptyPartVec);
       bulkData->declare_relation(elem, side,  4 /*local side id*/);

       stk::mesh::Entity llnode = bulkData->declare_entity(stk::topology::NODE_RANK, bf[i][1], nodePartVec);
       stk::mesh::Entity lrnode = bulkData->declare_entity(stk::topology::NODE_RANK, bf[i][2], nodePartVec);
       stk::mesh::Entity urnode = bulkData->declare_entity(stk::topology::NODE_RANK, bf[i][3], nodePartVec);
       stk::mesh::Entity ulnode = bulkData->declare_entity(stk::topology::NODE_RANK, bf[i][4], nodePartVec);
       
       bulkData->declare_relation(side, llnode, 0);
       bulkData->declare_relation(side, ulnode, 3);
       bulkData->declare_relation(side, urnode, 2);
       bulkData->declare_relation(side, lrnode, 1);
    }
    if (debug_output_verbosity != 0) *out << "...done." << std::endl;
  }
  if (have_tf == true) {
    if (debug_output_verbosity != 0) *out << "Setting top surface connectivity from data provided..." << std::endl;
    singlePartVec[0] = ssPartVec["Top"];
    for (int i=0; i<top_face_mapT->getNodeNumElements(); i++) {
       sideID = top_face_mapT->getGlobalElement(i);
       stk::mesh::EntityId side_id = (stk::mesh::EntityId)(sideID);
       stk::mesh::Entity side  = bulkData->declare_entity(metaData->side_rank(),side_id+1, singlePartVec);
       const unsigned int elem_GID = tf[i][0];
       stk::mesh::EntityId elem_id = (stk::mesh::EntityId) elem_GID;
       stk::mesh::Entity elem  = bulkData->declare_entity(stk::topology::ELEMENT_RANK, elem_id, emptyPartVec);
       bulkData->declare_relation(elem, side,  5 /*local side id*/);

       //TODO: check numbering convention! 
       stk::mesh::Entity llnode = bulkData->declare_entity(stk::topology::NODE_RANK, tf[i][1], nodePartVec);
       stk::mesh::Entity lrnode = bulkData->declare_entity(stk::topology::NODE_RANK, tf[i][2], nodePartVec);
       stk::mesh::Entity urnode = bulkData->declare_entity(stk::topology::NODE_RANK, tf[i][3], nodePartVec);
       stk::mesh::Entity ulnode = bulkData->declare_entity(stk::topology::NODE_RANK, tf[i][4], nodePartVec);
       
       bulkData->declare_relation(side, llnode, 0);
       bulkData->declare_relation(side, ulnode, 3);
       bulkData->declare_relation(side, urnode, 2);
       bulkData->declare_relation(side, lrnode, 1);
    }
    if (debug_output_verbosity != 0) *out << "...done." << std::endl;
  }

  //set lateral face connectivity
  if (have_wf == true) {
    if (debug_output_verbosity != 0) *out << "Setting west lateral surface connectivity from data provided..." << std::endl;
    singlePartVec[0] = ssPartVec["Lateral"];
    for (int i=0; i<west_face_mapT->getNodeNumElements(); i++) {
       sideID = west_face_mapT->getGlobalElement(i);
       stk::mesh::EntityId side_id = (stk::mesh::EntityId)(sideID);
       stk::mesh::Entity side  = bulkData->declare_entity(metaData->side_rank(),side_id+1, singlePartVec);
       const unsigned int elem_GID = wf[i][0];
       if (debug_output_verbosity != 0) *out << "   element " << elem_GID << " has a west lateral face." << std::endl; 
       stk::mesh::EntityId elem_id = (stk::mesh::EntityId) elem_GID;
       stk::mesh::Entity elem  = bulkData->declare_entity(stk::topology::ELEMENT_RANK, elem_id, emptyPartVec);
       bulkData->declare_relation(elem, side,  3 /*local side id*/);

       stk::mesh::Entity llnode = bulkData->declare_entity(stk::topology::NODE_RANK, wf[i][1], nodePartVec); //OK
       stk::mesh::Entity ulnode = bulkData->declare_entity(stk::topology::NODE_RANK, wf[i][2], nodePartVec); //OK
       stk::mesh::Entity ulnodeb = bulkData->declare_entity(stk::topology::NODE_RANK, wf[i][3], nodePartVec); //OK
       stk::mesh::Entity llnodeb = bulkData->declare_entity(stk::topology::NODE_RANK, wf[i][4], nodePartVec); //OK
       
       bulkData->declare_relation(side, llnode, 0);
       bulkData->declare_relation(side, llnodeb, 2);
       bulkData->declare_relation(side, ulnodeb, 3);
       bulkData->declare_relation(side, ulnode, 1);

    }
    if (debug_output_verbosity != 0) *out << "...done." << std::endl;
  }
  if (have_ef == true) {
    if (debug_output_verbosity != 0) *out << "Setting east lateral surface connectivity from data provided..." << std::endl;
    singlePartVec[0] = ssPartVec["Lateral"];
    for (int i=0; i<east_face_mapT->getNodeNumElements(); i++) {
       sideID = east_face_mapT->getGlobalElement(i);
       stk::mesh::EntityId side_id = (stk::mesh::EntityId)(sideID);
       stk::mesh::Entity side  = bulkData->declare_entity(metaData->side_rank(),side_id+1, singlePartVec);
       const unsigned int elem_GID = ef[i][0];
       if (debug_output_verbosity != 0) *out << "   element " << elem_GID << " has an east lateral face." << std::endl; 
       stk::mesh::EntityId elem_id = (stk::mesh::EntityId) elem_GID;
       stk::mesh::Entity elem  = bulkData->declare_entity(stk::topology::ELEMENT_RANK, elem_id, emptyPartVec);
       bulkData->declare_relation(elem, side,  1 /*local side id*/);

       stk::mesh::Entity lrnode = bulkData->declare_entity(stk::topology::NODE_RANK, ef[i][1], nodePartVec);
       stk::mesh::Entity lrnodeb = bulkData->declare_entity(stk::topology::NODE_RANK, ef[i][2], nodePartVec);
       stk::mesh::Entity urnodeb = bulkData->declare_entity(stk::topology::NODE_RANK, ef[i][3], nodePartVec);
       stk::mesh::Entity urnode = bulkData->declare_entity(stk::topology::NODE_RANK, ef[i][4], nodePartVec);
       
       bulkData->declare_relation(side, lrnode, 0);
       bulkData->declare_relation(side, urnode, 1);
       bulkData->declare_relation(side, urnodeb, 3);
       bulkData->declare_relation(side, lrnodeb, 2);
    }
    if (debug_output_verbosity != 0) *out << "...done." << std::endl;
  }
  if (have_sf == true) {
    if (debug_output_verbosity != 0) *out << "Setting south lateral surface connectivity from data provided..." << std::endl;
    singlePartVec[0] = ssPartVec["Lateral"];
    for (int i=0; i<south_face_mapT->getNodeNumElements(); i++) {
       sideID = south_face_mapT->getGlobalElement(i);
       stk::mesh::EntityId side_id = (stk::mesh::EntityId)(sideID);
       stk::mesh::Entity side  = bulkData->declare_entity(metaData->side_rank(),side_id+1, singlePartVec);
       const unsigned int elem_GID = sf[i][0];
       if (debug_output_verbosity != 0) *out << "   element " << elem_GID << " has a south lateral face." << std::endl; 
       stk::mesh::EntityId elem_id = (stk::mesh::EntityId) elem_GID;
       stk::mesh::Entity elem  = bulkData->declare_entity(stk::topology::ELEMENT_RANK, elem_id, emptyPartVec);
       bulkData->declare_relation(elem, side,  0 /*local side id*/);

       stk::mesh::Entity llnode = bulkData->declare_entity(stk::topology::NODE_RANK, sf[i][1], nodePartVec);
       stk::mesh::Entity lrnode = bulkData->declare_entity(stk::topology::NODE_RANK, sf[i][2], nodePartVec);
       stk::mesh::Entity lrnodeb = bulkData->declare_entity(stk::topology::NODE_RANK, sf[i][3], nodePartVec);
       stk::mesh::Entity llnodeb = bulkData->declare_entity(stk::topology::NODE_RANK, sf[i][4], nodePartVec);
       
       bulkData->declare_relation(side, llnode, 0);
       bulkData->declare_relation(side, lrnode, 1);
       bulkData->declare_relation(side, lrnodeb, 3);
       bulkData->declare_relation(side, llnodeb, 2);
    }
    if (debug_output_verbosity != 0) *out << "...done." << std::endl;
  }
  if (have_nf == true) {
    if (debug_output_verbosity != 0) *out << "Setting north lateral surface connectivity from data provided..." << std::endl;
    singlePartVec[0] = ssPartVec["Lateral"];
    for (int i=0; i<north_face_mapT->getNodeNumElements(); i++) {
       sideID = north_face_mapT->getGlobalElement(i);
       stk::mesh::EntityId side_id = (stk::mesh::EntityId)(sideID);
       stk::mesh::Entity side  = bulkData->declare_entity(metaData->side_rank(),side_id+1, singlePartVec);
       const unsigned int elem_GID = nf[i][0];
       if (debug_output_verbosity !=0) *out << "   element " << elem_GID << " has a north lateral face." << std::endl; 
       stk::mesh::EntityId elem_id = (stk::mesh::EntityId) elem_GID;
       stk::mesh::Entity elem  = bulkData->declare_entity(stk::topology::ELEMENT_RANK, elem_id, emptyPartVec);
       bulkData->declare_relation(elem, side,  2 /*local side id*/);

       stk::mesh::Entity ulnode = bulkData->declare_entity(stk::topology::NODE_RANK, nf[i][1], nodePartVec);
       stk::mesh::Entity ulnodeb = bulkData->declare_entity(stk::topology::NODE_RANK, nf[i][2], nodePartVec);
       stk::mesh::Entity urnodeb = bulkData->declare_entity(stk::topology::NODE_RANK, nf[i][3], nodePartVec);
       stk::mesh::Entity urnode = bulkData->declare_entity(stk::topology::NODE_RANK, nf[i][4], nodePartVec);
       
       bulkData->declare_relation(side, urnode, 0);
       bulkData->declare_relation(side, ulnode, 1);
       bulkData->declare_relation(side, ulnodeb, 3);
       bulkData->declare_relation(side, urnodeb, 2);
    }
    if (debug_output_verbosity != 0) *out << "...done." << std::endl;
  }

  Albany::fix_node_sharing(*bulkData);
  bulkData->modification_end();
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::CismSTKMeshStruct::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getValidGenericSTKParameters("Valid ASCII_DiscParams");

  return validPL;
}
