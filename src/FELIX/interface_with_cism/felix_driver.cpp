
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

//uncomment the following if you want to write stuff out to matrix market to debug
//#define WRITE_TO_MATRIX_MARKET 

#ifdef WRITE_TO_MATRIX_MARKET
#include "EpetraExt_MultiVectorOut.h"
#include "EpetraExt_BlockMapOut.h"
#endif 

Teuchos::RCP<Albany::CismSTKMeshStruct> meshStruct;
Teuchos::RCP<const Epetra_Comm> mpiComm;
Teuchos::RCP<Teuchos::ParameterList> appParams;
Teuchos::RCP<Teuchos::ParameterList> discParams;
Teuchos::RCP<Albany::SolverFactory> slvrfctry;
Teuchos::RCP<Thyra::ModelEvaluator<double> > solver;
//IK, 11/14/13: what is reducedComm for? 
MPI_Comm comm, reducedComm;
bool interleavedOrdering; 


int rank, number_procs;
long  cism_communicator; 
int cism_process_count, my_cism_rank;
double dew, dns;
long * dimInfo;        
int * dimInfoGeom; 
long ewlb, ewub, nslb, nsub;
long ewn, nsn, upn, nhalo; 
long global_ewn, global_nsn; 
double * seconds_per_year_ptr, * gravity_ptr, * rho_ice_ptr, * rho_seawater_ptr;
double * thicknessDataPtr, *topographyDataPtr;
double * upperSurfaceDataPtr, * lowerSurfaceDataPtr;
double * floating_maskDataPtr, * ice_maskDataPtr, * lower_cell_locDataPtr;
long nCellsActive;
int nNodes, nElementsActive;  
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
Teuchos::RCP<Epetra_Map> node_map; 

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
    global_ewn = dimInfoGeom[2]; 
    global_nsn = dimInfoGeom[3]; 
    std::cout << "In felix_driver: global_ewn = " << global_ewn << ", global_nsn = " << global_nsn << std::endl;
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

    nNodes = (ewn-2*nhalo+1)*(nsn-2*nhalo+1)*upn; //number of nodes in mesh (on each processor) 
    nElementsActive = nCellsActive*(upn-1); //number of 3D active elements in mesh  
    
    meshStruct = Teuchos::rcp(new Albany::CismSTKMeshStruct(discParams, mpiComm, xyz_at_nodes_Ptr, global_node_id_owned_map_Ptr, global_element_id_active_owned_map_Ptr, 
                                                           global_element_conn_active_Ptr, global_basal_face_id_active_owned_map_Ptr, global_basal_face_conn_active_Ptr, 
                                                           beta_at_nodes_Ptr, surf_height_at_nodes_Ptr, flwa_at_active_elements_Ptr, nNodes, nElementsActive, nCellsActive));
    meshStruct->constructMesh(mpiComm, discParams, neq, req, sis, meshStruct->getMeshSpecs()[0]->worksetSize);
 
    interleavedOrdering = meshStruct->getInterleavedOrdering();
    Albany::AbstractSTKFieldContainer::VectorFieldType* solutionField;
    if(interleavedOrdering)
      solutionField = Teuchos::rcp_dynamic_cast<Albany::OrdinarySTKFieldContainer<true> >(meshStruct->getFieldContainer())->getSolutionField();
    else
      solutionField = Teuchos::rcp_dynamic_cast<Albany::OrdinarySTKFieldContainer<false> >(meshStruct->getFieldContainer())->getSolutionField();

    //Set restart solution to the one passed from CISM 
    //TO DO: Set initial condition to uvel and vvel from Glimmer/CISM.  This is not being done yet.  
    //Need to do something special for interleavedOrdering = false case here?
    //global_node_id_owned_map_Ptr is 1-based, so node_map is 1-based
    node_map = Teuchos::rcp(new Epetra_Map(-1, nNodes, global_node_id_owned_map_Ptr, 0, *mpiComm));
    //TO DO: pass only active nodes from Glimmer-CISM? 
    /*for (int i=0; i<nElementsActive; i++) {
      for (int j=0; j<8; j++) {
        int node_GID =  global_element_conn_active_Ptr[i + nElementsActive*j]; //node_GID is 1-based
        int node_LID =  node_map->LID(node_GID); //node_LID is 0-based 
        stk::mesh::Entity& node = *meshStruct->bulkData->get_entity(meshStruct->metaData->node_rank(), node_GID);
        double* sol = stk::mesh::field_data(*solutionField, node);
        //std::cout << "uVel_ptr: " << uVel_ptr[node_LID] << std::endl; 
        sol[0] = uVel_ptr[node_LID];
        sol[1] = vVel_ptr[node_LID];
     }
   }*/
 
    // clean up
    std::cout << "exec mode = " << exec_mode << std::endl;

    std::cout << "End of nested scope." << std::endl; 
  }  

 

}

// The solve is done in the felix_driver_run function, and the solution is passed back to Glimmer-CISM 
// IK, 12/3/13: time_inc_yr and cur_time_yr are not used here... 
void felix_driver_run(FelixToGlimmer * ftg_ptr, double& cur_time_yr, double time_inc_yr)
{
  
    std::cout << "In felix_driver_run, cur_time, time_inc = " << cur_time_yr << "   " << time_inc_yr << std::endl;
    // ---------------------------------------------------------------------------------------------------
    // Solve 
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

     const Epetra_Map& ownedMap(*app->getDiscretization()->getMap()); //owned map
     const Epetra_Map& overlapMap(*app->getDiscretization()->getOverlapMap()); //overlap map
     Epetra_Import import(overlapMap, ownedMap); //importer from ownedMap to overlapMap 
     Epetra_Vector solutionOverlap(overlapMap); //overlapped solution 
     solutionOverlap.Import(*app->getDiscretization()->getSolutionField(), import, Insert);

#ifdef WRITE_TO_MATRIX_MARKET
    //For debug: write solution and maps to matrix market file 
     EpetraExt::BlockMapToMatrixMarketFile("node_map.mm", *node_map); 
     EpetraExt::BlockMapToMatrixMarketFile("map.mm", ownedMap); 
     EpetraExt::BlockMapToMatrixMarketFile("overlap_map.mm", overlapMap); 
     EpetraExt::MultiVectorToMatrixMarketFile("solution.mm", *app->getDiscretization()->getSolutionField());
#endif
    // ---------------------------------------------------------------------------------------------------
    // Copy solution back to glimmer uvel and vvel arrays to be passed back
    // ---------------------------------------------------------------------------------------------------

    //std::cout << "overlapMap # global elements: " << overlapMap.NumGlobalElements() << std::endl; 
    //std::cout << "overlapMap # my elements: " << overlapMap.NumMyElements() << std::endl; 
    //std::cout << "overlapMap: " << overlapMap << std::endl; 
    //std::cout << "map # global elements: " << ownedMap.NumGlobalElements() << std::endl; 
    //std::cout << "map # my elements: " << ownedMap.NumMyElements() << std::endl; 
    //std::cout << "node_map # global elements: " << node_map->NumGlobalElements() << std::endl; 
    //std::cout << "node_map # my elements: " << node_map->NumMyElements() << std::endl; 
    //std::cout << "node_map: " << *node_map << std::endl; 

    //Epetra_Vectors to hold uvel and vvel to be passed to Glimmer/CISM 
    Epetra_Vector uvel(*node_map, true); 
    Epetra_Vector vvel(*node_map, true); 

    
    if (interleavedOrdering == true) { 
      for (int i=0; i<overlapMap.NumMyElements(); i++) { 
        int global_dof = overlapMap.GID(i);
        double sol_value = solutionOverlap[i];  
        int modulo = (global_dof % 2); //check if dof is for u or for v 
        int vel_global_dof, vel_local_dof; 
        if (modulo == 0) { //u dof 
          vel_global_dof = global_dof/2+1; //add 1 because node_map is 1-based 
          vel_local_dof = node_map->LID(vel_global_dof); //look up local id corresponding to global id in node_map
          //std::cout << "uvel: global_dof = " << global_dof << ", uvel_global_dof = " << vel_global_dof << ", uvel_local_dof = " << vel_local_dof << std::endl; 
          uvel.ReplaceMyValues(1, &sol_value, &vel_local_dof); 
        }
        else { // v dof 
          vel_global_dof = (global_dof-1)/2+1; //add 1 because node_map is 1-based 
          vel_local_dof = node_map->LID(vel_global_dof); //look up local id corresponding to global id in node_map
          vvel.ReplaceMyValues(1, &sol_value, & vel_local_dof); 
        }
      }
    }
    else { //note: the case with non-interleaved ordering has not been tested...
      int numDofs = overlapMap.NumGlobalElements(); 
      for (int i=0; i<overlapMap.NumMyElements(); i++) { 
        int global_dof = overlapMap.GID(i);
        double sol_value = solutionOverlap[i];  
        int vel_global_dof, vel_local_dof; 
        if (global_dof < numDofs/2) { //u dof
          vel_global_dof = global_dof+1; //add 1 because node_map is 1-based 
          vel_local_dof = node_map->LID(vel_global_dof); //look up local id corresponding to global id in node_map
          uvel.ReplaceMyValues(1, &sol_value, &vel_local_dof); 
        }
        else { //v dofs 
          vel_global_dof = global_dof-numDofs/2+1; //add 1 because node_map is 1-based
          vel_local_dof = node_map->LID(vel_global_dof); //look up local id corresponding to global id in node_map
          vvel.ReplaceMyValues(1, &sol_value, & vel_local_dof);
        } 
      }
    }
 

#ifdef WRITE_TO_MATRIX_MARKET
    //For debug: write solution to matrix market file 
     EpetraExt::MultiVectorToMatrixMarketFile("uvel.mm", uvel); 
     EpetraExt::MultiVectorToMatrixMarketFile("vvel.mm", vvel);
#endif
 
     //create vector used to renumber nodes on each processor from the Albany convention (horizontal levels first) to the CISM convention (vertical layers first)
     int nNodes2D = (global_ewn + 1)*(global_nsn+1); //number global nodes in the domain in 2D 
     int nNodesProc2D = (nsn-2*nhalo+1)*(ewn-2*nhalo+1); //number of nodes on each processor in 2D  
     std::vector<int> cismToAlbanyNodeNumberMap(upn*nNodesProc2D);

     for (int j=0; j<nsn-2*nhalo+1;j++) { 
       for (int i=0; i<ewn-2*nhalo+1; i++) {
         for (int k=0; k<upn; k++) { 
           int index = k+upn*i + j*(ewn-2*nhalo+1)*upn; 
           cismToAlbanyNodeNumberMap[index] = k*nNodes2D + global_node_id_owned_map_Ptr[i+j*(ewn-2*nhalo+1)]; 
           //if (mpiComm->MyPID() == 0) 
           //  std::cout << "index: " << index << ", cismToAlbanyNodeNumberMap: " << cismToAlbanyNodeNumberMap[index] << std::endl; 
          }
        }
     }

     //Copy uvel and vvel into uVel_ptr and vVel_ptr respectively (the arrays passed back to CISM) according to the numbering consistent w/ CISM. 
     int tmp = 0; 
     int tmp1 = 0; 
     int local_nodeID;  
     for (int j=0; j<nsn-1; j++) {
       for (int i=0; i<ewn-1; i++) { 
         for (int k=0; k<upn; k++) {
           if (j >= nhalo-1 & j < nsn-nhalo) {
             if (i >= nhalo-1 & i < ewn-nhalo) { 
               local_nodeID = node_map->LID(cismToAlbanyNodeNumberMap[tmp]); 
               //if (mpiComm->MyPID() == 0) 
               //std::cout << "tmp:" << tmp << ", cismToAlbanyNodeNumberMap[tmp]: " << cismToAlbanyNodeNumberMap[tmp] << ", local_nodeID: " 
               //<< local_nodeID << ", uvel: " << uvel[local_nodeID] << std::endl; //uvel[local_nodeID] << std::endl;  
               uVel_ptr[tmp1] = uvel[local_nodeID];
               vVel_ptr[tmp1] = vvel[local_nodeID];  
               tmp++;
            }
            }
            else {
             uVel_ptr[tmp1] = 0.0; 
             vVel_ptr[tmp1] = 0.0; 
            }
            tmp1++; 
         }
        }
      }


    first_time_step = false;
 
}
  

//Clean up
//IK, 12/3/13: this is not called anywhere in the interface code...  used to be called (based on old bisicles interface code)?  
void felix_driver_finalize(int amr_obj_index)
{

  std::cout << "In felix_driver_finalize: cleaning up..." << std::endl;
  std::cout << "done cleaning up!" << std::endl << std::endl; 
  
}

