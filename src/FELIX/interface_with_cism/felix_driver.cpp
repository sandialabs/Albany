
#include <iostream>
#include <fstream>
#include "felix_driver.H"
#include "Albany_CismSTKMeshStruct.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Albany_Utils.hpp"
#include "Albany_SolverFactory.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include <stk_mesh/base/FieldData.hpp>
#include "Piro_PerformSolve.hpp"
#include "Thyra_EpetraThyraWrappers.hpp"
#include <stk_io/IossBridge.hpp>
#include <stk_io/MeshReadWriteUtils.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/FieldData.hpp>
#include <Ionit_Initializer.h>
#include "Albany_OrdinarySTKFieldContainer.hpp"
#include "EpetraExt_MultiVectorOut.h"

Teuchos::RCP<Albany::CismSTKMeshStruct> meshStruct;
Teuchos::RCP<const Epetra_Comm> mpiComm;
Teuchos::RCP<Teuchos::ParameterList> appParams;
Teuchos::RCP<Teuchos::ParameterList> discParams;
Teuchos::RCP<Albany::SolverFactory> slvrfctry;
Teuchos::RCP<Thyra::ModelEvaluator<double> > solver;
//IK, 11/14/13: what is reducedComm for? 
MPI_Comm comm, reducedComm;


int rank, number_procs;
long  cism_communicator; 
int cism_process_count, my_cism_rank;
double dew, dns;
//need to delete these things in cleanup...
long * dimInfo;        
int * dimInfoGeom; 
long ewlb, ewub, nslb, nsub;
long ewn, nsn, upn, nhalo; 
double * seconds_per_year_ptr, * gravity_ptr, * rho_ice_ptr, * rho_seawater_ptr;
double * thicknessDataPtr, *topographyDataPtr;
double * upperSurfaceDataPtr, * lowerSurfaceDataPtr;
double * floating_maskDataPtr, * ice_maskDataPtr, * lower_cell_locDataPtr;
long nCellsActive; 
double* xyz_at_nodes_Ptr, *surf_height_at_nodes_Ptr, *beta_at_nodes_Ptr;
double *flwa_at_active_elements_Ptr; 
int * global_node_id_owned_map_Ptr; 
int * global_element_conn_active_Ptr; 
int * global_element_id_active_owned_map_Ptr; 
int * global_basal_face_conn_active_Ptr; 
int * global_basal_face_id_active_owned_map_Ptr; 
double *uVel_ptr; 
double *vVel_ptr; 
bool first_time_step = true; 

extern "C" void felix_driver_();

//What is exec_mode??
void felix_driver_init(int argc, int exec_mode, FelixToGlimmer * ftg_ptr, const char * input_fname)
{ 

  std::cout << "In felix_driver..." << std::endl;
  std::cout << "Printing this from Albany...  This worked!  Yay!  Nov. 14 2013" << std::endl; 


  { // Begin nested scope
    
   std::cout << "Beginning nested scope..." << std::endl;

    // ---------------------------------------------
    //get communicator / communicator info from CISM
    //TO DO: ifdef to check if CISM and Albany have MPI?  
    //#ifdef HAVE_MPI
    //#else
    //#endif
    // ---------------------------------------------
    // The following line needs to change...  It is for a serial run...
    cism_communicator = *(ftg_ptr -> getLongVar("communicator","mpi_vars"));
    cism_process_count = *(ftg_ptr -> getLongVar("process_count","mpi_vars"));
    my_cism_rank = *(ftg_ptr -> getLongVar("my_rank","mpi_vars"));
    //get MPI_COMM from Fortran
    comm = MPI_Comm_f2c(cism_communicator);
    std::cout << "after MPI_Comm_f2c!" << std::endl;  
    //MPI_COMM_size (comm, &cism_process_count); 
    //MPI_COMM_rank (comm, &my_cism_rank); 
    //convert comm to Epetra_Comm 
    //mpiComm = Albany::createEpetraCommFromMpiComm(reducedComm); 
    mpiComm = Albany::createEpetraCommFromMpiComm(comm); 
    std::cout << "after createEpetraCommFromMpiComm!" << std::endl;  


    // ---------------------------------------------
    // get geometry info from CISM  
    //IK, 11/14/13: these things may not be needed in Albany/FELIX...  for now they are passed anyway.
    // ---------------------------------------------
    
    std::cout << "Getting geometry info from CISM..." << std::endl; 
    dimInfo = ftg_ptr -> getLongVar("dimInfo","geometry");
    dew = *(ftg_ptr -> getDoubleVar("dew","numerics"));
    dns = *(ftg_ptr -> getDoubleVar("dns","numerics"));
    std::cout << "In felix_driver: dew, dns = " << dew << "  " << dns << std::endl;
    dimInfoGeom = new int[dimInfo[0]+1];    
    for (int i=0;i<=dimInfo[0];i++) dimInfoGeom[i] = dimInfo[i];   
    std::cout << "DimInfoGeom  in felix_driver: " << std::endl;
    for (int i=0;i<=dimInfoGeom[0];i++) std::cout << dimInfoGeom[i] << " ";
    std::cout << std::endl;
    ewlb = *(ftg_ptr -> getLongVar("ewlb","geometry"));
    ewub = *(ftg_ptr -> getLongVar("ewub","geometry"));
    nslb = *(ftg_ptr -> getLongVar("nslb","geometry"));
    nsub = *(ftg_ptr -> getLongVar("nsub","geometry"));
    nhalo = *(ftg_ptr -> getLongVar("nhalo","geometry"));
    ewn = *(ftg_ptr -> getLongVar("ewn","geometry"));
    nsn = *(ftg_ptr -> getLongVar("nsn","geometry"));
    upn = *(ftg_ptr -> getLongVar("upn","geometry"));
    std::cout << "In felix_driver: ewn = " << ewn << ", nsn = " << nsn << ", upn = " << upn << ", nhalo = " << nhalo << std::endl;


    // ---------------------------------------------
    // get constants from CISM
    // IK, 11/14/13: these things may not be needed in Albany/FELIX...  for now they are passed anyway.
    // ---------------------------------------------
 
    seconds_per_year_ptr = ftg_ptr -> getDoubleVar("seconds_per_year","constants");
    gravity_ptr = ftg_ptr -> getDoubleVar("gravity","constants");
    rho_ice_ptr = ftg_ptr -> getDoubleVar("rho_ice","constants");
    rho_seawater_ptr = ftg_ptr -> getDoubleVar("rho_seawater","constants");
    thicknessDataPtr = ftg_ptr -> getDoubleVar("thck","geometry");
    topographyDataPtr = ftg_ptr -> getDoubleVar("topg","geometry");
    upperSurfaceDataPtr = ftg_ptr -> getDoubleVar("usrf","geometry");
    lowerSurfaceDataPtr = ftg_ptr -> getDoubleVar("lsrf","geometry");
    floating_maskDataPtr = ftg_ptr -> getDoubleVar("floating_mask","geometry");
    ice_maskDataPtr = ftg_ptr -> getDoubleVar("ice_mask","geometry");
    lower_cell_locDataPtr = ftg_ptr -> getDoubleVar("lower_cell_loc","geometry");

    // ---------------------------------------------
    // get connectivity arrays from CISM 
    // IK, 11/14/13: these things may not be needed in Albany/FELIX...  for now they are passed anyway.
    // ---------------------------------------------
    std::cout << "In felix_driver: grabbing connectivity array pointers from CISM..." << std::endl; 
    //IK, 11/13/13: check that connectivity derived types are transfered over from CISM to Albany/FELIX    
    nCellsActive = *(ftg_ptr -> getLongVar("nCellsActive","connectivity")); 
    std::cout << "In felix_driver: nCellsActive = " << nCellsActive <<  std::endl;
    xyz_at_nodes_Ptr = ftg_ptr -> getDoubleVar("xyz_at_nodes","connectivity"); 
    surf_height_at_nodes_Ptr = ftg_ptr -> getDoubleVar("surf_height_at_nodes","connectivity"); 
    beta_at_nodes_Ptr = ftg_ptr -> getDoubleVar("beta_at_nodes","connectivity");
    flwa_at_active_elements_Ptr = ftg_ptr -> getDoubleVar("flwa_at_active_elements","connectivity"); 
    global_node_id_owned_map_Ptr = ftg_ptr -> getInt4Var("global_node_id_owned_map","connectivity");  
    global_element_conn_active_Ptr = ftg_ptr -> getInt4Var("global_element_conn_active","connectivity");  
    global_element_id_active_owned_map_Ptr = ftg_ptr -> getInt4Var("global_element_id_active_owned_map","connectivity");  
    global_basal_face_conn_active_Ptr = ftg_ptr -> getInt4Var("global_basal_face_conn_active","connectivity");  
    global_basal_face_id_active_owned_map_Ptr = ftg_ptr -> getInt4Var("global_basal_face_id_active_owned_map","connectivity");  
    std::cout << "...done!" << std::endl; 

    // ---------------------------------------------
    // get u and v velocity solution from Glimmer-CISM 
    // IK, 11/26/13: need to concatenate these into a single solve for initial condition for Albany/FELIX solve  
    // ---------------------------------------------
    std::cout << "In felix_driver: grabbing pointers to u and v velocities in CISM..." << std::endl; 
    uVel_ptr = ftg_ptr ->getDoubleVar("uvel", "velocity"); 
    vVel_ptr = ftg_ptr ->getDoubleVar("vvel", "velocity"); 
    std::cout << "...done!" << std::endl; 
    


    // ---------------------------------------------
    // create Albany mesh  
    // ---------------------------------------------
    slvrfctry = Teuchos::rcp(new Albany::SolverFactory("input_albany-cism.xml", comm));
    discParams = Teuchos::sublist(Teuchos::rcp(&slvrfctry->getParameters(),false), "Discretization", true);
    Teuchos::RCP<Albany::StateInfoStruct> sis=Teuchos::rcp(new Albany::StateInfoStruct);
    Albany::AbstractFieldContainer::FieldContainerRequirements req;
    req.push_back("Surface Height");
    req.push_back("Temperature");
    req.push_back("Basal Friction");
    req.push_back("Thickness");
    req.push_back("Flow Factor");
    int neq = 2; //number of equations - 2 for FO Stokes
    //IK, 11/14/13, debug output: check that pointers that are passed from CISM are not null 
    std::cout << "DEBUG: xyz_at_nodes_Ptr:" << xyz_at_nodes_Ptr << std::endl; 
    std::cout << "DEBUG: surf_height_at_nodes_Ptr:" << surf_height_at_nodes_Ptr << std::endl; 
    std::cout << "DEBUG: beta_at_nodes_Ptr:" << beta_at_nodes_Ptr << std::endl; 
    std::cout << "DEBUG: flwa_at_active_elements_Ptr:" << flwa_at_active_elements_Ptr << std::endl; 
    std::cout << "DEBUG: global_node_id_owned_map_Ptr:" << global_node_id_owned_map_Ptr << std::endl; 
    std::cout << "DEBUG: global_element_conn_active_Ptr:" << global_element_conn_active_Ptr << std::endl; 
    std::cout << "DEBUG: global_basal_face_conn_active_Ptr:" << global_basal_face_conn_active_Ptr << std::endl; 
    std::cout << "DEBUG: global_basal_face_id_active_owned_map_Ptr:" << global_basal_face_id_active_owned_map_Ptr << std::endl;

    int nNodes = (ewn-2*nhalo+1)*(nsn-2*nhalo+1)*upn; //number of nodes in mesh
    int nElementsActive = nCellsActive*(upn-1); //number of 3D active elements in mesh  
    
    meshStruct = Teuchos::rcp(new Albany::CismSTKMeshStruct(discParams, mpiComm, xyz_at_nodes_Ptr, global_node_id_owned_map_Ptr, global_element_id_active_owned_map_Ptr, 
                                                           global_element_conn_active_Ptr, global_basal_face_id_active_owned_map_Ptr, global_basal_face_conn_active_Ptr, 
                                                           beta_at_nodes_Ptr, surf_height_at_nodes_Ptr, flwa_at_active_elements_Ptr, nNodes, nElementsActive, nCellsActive));
    meshStruct->constructMesh(mpiComm, discParams, neq, req, sis, meshStruct->getMeshSpecs()[0]->worksetSize);
 
    const bool interleavedOrdering = meshStruct->getInterleavedOrdering();
    Albany::AbstractSTKFieldContainer::VectorFieldType* solutionField;
    if(interleavedOrdering)
      solutionField = Teuchos::rcp_dynamic_cast<Albany::OrdinarySTKFieldContainer<true> >(meshStruct->getFieldContainer())->getSolutionField();
    else
      solutionField = Teuchos::rcp_dynamic_cast<Albany::OrdinarySTKFieldContainer<false> >(meshStruct->getFieldContainer())->getSolutionField();

    //Set restart solution to the one passed from CISM 
    //TO DO: this is not quite right!  Need to figure out how uvel and vvel are organized.  Do these include non-active nodes? 
    //Need to do something special for interleavedOrdering = false case here?
    //global_node_id_owned_map_Ptr is 1-based, so node_map is 1-based
    Teuchos::RCP<Epetra_Map> node_map = Teuchos::rcp(new Epetra_Map(-1, nNodes, global_node_id_owned_map_Ptr, 0, *mpiComm));
    //TO DO: pass only active nodes from Glimmer-CISM.  Then this loop can be over the active nodes rather than the active elements, which is redundant. 
    for (int i=0; i<nElementsActive; i++) {
      for (int j=0; j<8; j++) {
        int node_GID =  global_element_conn_active_Ptr[i + nElementsActive*j]; //node_GID is 1-based
        int node_LID =  node_map->LID(node_GID); //node_LID is 0-based 
        stk::mesh::Entity& node = *meshStruct->bulkData->get_entity(meshStruct->metaData->node_rank(), node_GID);
        double* sol = stk::mesh::field_data(*solutionField, node);
        //std::cout << "uVel_ptr: " << uVel_ptr[node_LID] << std::endl; 
        sol[0] = uVel_ptr[node_LID];
        sol[1] = vVel_ptr[node_LID];
     }
   }
 

    // ---------------------------------------------------------------------------------------------------
    // Solve 
    // IK, 11/26/13, TO DO: move to a separate function?  
    // ---------------------------------------------------------------------------------------------------

    //Need to set HasRestart solution such that uvel_Ptr and vvel_Ptr (u and v from Glimmer/CISM) are always set as initial condition?  
    meshStruct->setHasRestartSolution(!first_time_step);
 
    Teuchos::RCP<Albany::AbstractSTKMeshStruct> stkMeshStruct = meshStruct;
    discParams->set("STKMeshStruct",stkMeshStruct);
    Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(&slvrfctry->getParameters(),false);
    //TO DO: add checking if first time step or not
    if (!first_time_step)
    {
       meshStruct->setRestartDataTime(paramList->sublist("Problem").get("Homotopy Restart Step", 1.));
       double homotopy = paramList->sublist("Problem").sublist("FELIX Viscosity").get("Glen's Law Homotopy Parameter", 1.0);
       if(meshStruct->restartDataTime()== homotopy)
         paramList->sublist("Problem").set("Solution Method", "Steady");
    }
    Teuchos::RCP<Albany::Application> app = Teuchos::rcp(new Albany::Application(mpiComm, paramList));
    solver = slvrfctry->createThyraSolverAndGetAlbanyApp(app, mpiComm, mpiComm);


    Teuchos::ParameterList solveParams;
    solveParams.set("Compute Sensitivities", false);
    Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<double> > > thyraResponses;
    Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Thyra::MultiVectorBase<double> > > > thyraSensitivities;
    Piro::PerformSolveBase(*solver, solveParams, thyraResponses, thyraSensitivities);

    const Epetra_Map& overlapMap(*app->getDiscretization()->getOverlapMap());
    Epetra_Import import(overlapMap, *app->getDiscretization()->getMap());
    Epetra_Vector solution(overlapMap);
    solution.Import(*app->getDiscretization()->getSolutionField(), import, Insert);

    // ---------------------------------------------------------------------------------------------------
    // Copy solution back to glimmer uvel and vvel arrays to be passed back
    // TO DO: this is not quite right!  Need to figure out how uvel and vvel are organized in Glimmer/CISM  
    // IK, 11/27/13, TO DO: move to a separate function?  
    // ---------------------------------------------------------------------------------------------------

    std::cout << "overlapMap # global elements: " << overlapMap.NumGlobalElements() << std::endl; 
    std::cout << "overlapMap # my elements: " << overlapMap.NumMyElements() << std::endl; 
    std::cout << "overlapMap: " << overlapMap << std::endl; 
    std::cout << "node_map # global elements: " << node_map->NumGlobalElements() << std::endl; 
    std::cout << "node_map # my elements: " << node_map->NumMyElements() << std::endl; 
    std::cout << "node_map: " << *node_map << std::endl; 

    if (interleavedOrdering == true) { //default in Albany is inteleaved: solution is u, v, u, v, etc.
      for (int i=0; i<nElementsActive; i++) {
        for (int j=0; j<8; j++) {
          int node_GID =  global_element_conn_active_Ptr[i + nElementsActive*j]; //node_GID is 1-based
          int node_LID = node_map->LID(node_GID); //node_map is 1-based 
          int node_LID_uVel = overlapMap.LID(neq*(node_GID-1)); //overlapMap is 0-based 
          int node_LID_vVel = node_LID_uVel + 1; 
          uVel_ptr[node_LID] = solution[node_LID_uVel]; 
          vVel_ptr[node_LID] = solution[node_LID_vVel];    
        }
      }
   }
   else {
      int nActiveNodes = (overlapMap.NumGlobalElements())/2; 
      for (int i=0; i<nElementsActive; i++) {
        for (int j=0; j<8; j++) {
          int node_GID =  global_element_conn_active_Ptr[i + nElementsActive*j]; //node_GID is 1-based
          int node_LID = node_map->LID(node_GID); //node_map is 1-based 
          int node_LID_uVel = overlapMap.LID(node_GID-1); //overlapMap is 0-based 
          int node_LID_vVel = node_LID_uVel + nActiveNodes; 
          uVel_ptr[node_LID] = solution[node_LID_uVel]; 
          vVel_ptr[node_LID] = solution[node_LID_vVel];    
        }
      }
   }
   //For debug: write solution to matrix market file 
   EpetraExt::MultiVectorToMatrixMarketFile("solution.mm", solution); 

  /* else {
      }

      for ( UInt j = 0 ; j < numVertices3D ; ++j )
                   {
                           int ib = (Ordering == 0)*(j%lVertexColumnShift) + (Ordering == 1)*(j/vertexLayerShift);
                           int il = (Ordering == 0)*(j/lVertexColumnShift) + (Ordering == 1)*(j%vertexLayerShift);
                           int gId = il*vertexColumnShift+vertexLayerShift * indexToVertexID[ib];

                           int lId0, lId1;

                           if(interleavedOrdering)
                           {
                                   lId0= overlapMap.LID(2*gId);
                                   lId1 = lId0+1;
                           }
                           else
                           {
                                   lId0 = overlapMap.LID(gId);
                                   lId1 = lId0+numVertices3D;
                           }
                           velocityOnVertices[j] = solution[lId0];
                           velocityOnVertices[j + numVertices3D] = solution[lId1];
                   }

                   std::vector<int> mpasIndexToVertexID (nVertices);
                   for (int i = 0; i < nVertices; i++)
                   {
                           mpasIndexToVertexID[i] = indexToCellID_F[vertexToFCell[i]];
                   }
                   get_tetraP1_velocity_on_FEdges (u_normal_F, velocityOnVertices, edgeToFEdge, mpasIndexToVertexID);
                }

*/

    first_time_step = false;


    // clean up
    std::cout << "exec mode = " << exec_mode << std::endl;

    std::cout << "End of nested scope." << std::endl; 
  }  

 

}


// updates cur_time_yr as solution is advanced
// IK, 11/27/13: what should happen here??  Solve? 
void felix_driver_run(FelixToGlimmer * ftg_ptr, float& cur_time_yr, float time_inc_yr)
{
  
  std::cout << "In felix_driver_run, cur_time, time_inc = " 
       << cur_time_yr << "   " << time_inc_yr << std::endl;
 
}
  

//Clean up
//Should this be done here or in felix_driver_init?  
void felix_driver_finalize(int amr_obj_index)
{

  std::cout << "In felix_driver_finalize: cleaning up..." << std::endl;
  std::cout << "done cleaning up!" << std::endl << std::endl; 
//MPI_Finalize();
  
}

