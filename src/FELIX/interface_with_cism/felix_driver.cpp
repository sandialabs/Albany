
#include <iostream>
#include <fstream>
#include "felix_driver.H"
//#include "Albany_MpasSTKMeshStruct.hpp"
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



//Teuchos::RCP<Albany::MpasSTKMeshStruct> meshStruct2D;
//Teuchos::RCP<Albany::MpasSTKMeshStruct> meshStruct;
Teuchos::RCP<const Epetra_Comm> mpiComm;
Teuchos::RCP<Teuchos::ParameterList> appParams;
Teuchos::RCP<Teuchos::ParameterList> discParams;
Teuchos::RCP<Albany::SolverFactory> slvrfctry;
Teuchos::RCP<Thyra::ModelEvaluator<double> > solver;
//IK, 11/14/13: what is reducedComm for? 
MPI_Comm comm, reducedComm;


int rank, number_procs;
long cism_communicator, cism_process_count, my_cism_rank;
double dew, dns;
//need to delete these things in cleanup...
long * dimInfo;        
int * dimInfoGeom; 
long ewlb, ewub, nslb, nsub;
int ewn, nsn; 
double * seconds_per_year_ptr, * gravity_ptr, * rho_ice_ptr, * rho_seawater_ptr;
double * thicknessDataPtr, *topographyDataPtr;
double * upperSurfaceDataPtr, * lowerSurfaceDataPtr;
double * floating_maskDataPtr, * ice_maskDataPtr, * lower_cell_locDataPtr;
long nCellsActive; 
double* xyz_at_nodes_Ptr, *surf_height_at_nodes_Ptr, *beta_at_nodes_Ptr;
double *flwa_at_active_elements_Ptr; 
long int* global_node_id_owned_map_Ptr; 
long int* global_element_conn_active_Ptr; 
long int* global_element_id_active_owned_map_Ptr; 
long int* global_basal_face_conn_active_Ptr; 
long int* global_basal_face_id_active_owned_map_Ptr; 

extern "C" void felix_driver_();

int
felix_store(int obj_index, Felix ** felix_object, int mode)
{
  static Felix * felix_store_ptr_arr[DYCORE_MODEL_COUNT];
  //what happens here? 
  switch (mode) {
  case 0: felix_store_ptr_arr[obj_index] = *felix_object;
    std::cout << "In felix_store, mode = 0 -- Storing Felix Object # " 
	 << obj_index << ", Address = " << *felix_object << std::endl;
    break;
  case 1: *felix_object = felix_store_ptr_arr[obj_index];
    std::cout << "In felix_store, mode = 1 -- Retrieving Felix Object # " 
	 << obj_index << ", Address = " << *felix_object << std::endl;
    break;
  default: ;
  }
  return 0;
}

//What is exec_mode??
void felix_driver_run(int argc, int exec_mode);
 
void felix_driver_init(int argc, int exec_mode,FelixToGlimmer * btg_ptr, const char * input_fname)
{ 

  std::cout << "In felix_driver..." << std::endl;
  std::cout << "Printing this from Albany...  This worked!  Yay!  Nov. 14 2013" << std::endl; 


  { // Begin nested scope
    
  std::cout << "Beginning nested scope..." << std::endl;

    // ---------------------------------------------
    //get communicator / communicator info from CISM 
    // ---------------------------------------------
    cism_communicator = *(btg_ptr -> getLongVar("communicator","mpi_vars"));
    cism_process_count = *(btg_ptr -> getLongVar("process_count","mpi_vars"));
    my_cism_rank = *(btg_ptr -> getLongVar("my_rank","mpi_vars"));
    std::cout << "In felix_driver, CISM comm, count, my_rank = " << cism_communicator << "  "
         << cism_process_count << "  " << my_cism_rank << std::endl;
    std::cout << "rank " << rank << " of " << number_procs << std::endl;
    //get MPI_COMM from Fortran
    comm = MPI_Comm_f2c(cism_communicator);  

    //What is felixPtr for? 
    Felix* felixPtr = new Felix();


    // ---------------------------------------------
    // get geometry info from CISM  
    //IK, 11/14/13: these things may not be needed in Albany/FELIX...  for now they are passed anyway.
    // ---------------------------------------------
    
    std::cout << "Getting geometry info from CISM..." << std::endl; 
    dimInfo = btg_ptr -> getLongVar("dimInfo","geometry");
    dew = *(btg_ptr -> getDoubleVar("dew","numerics"));
    dns = *(btg_ptr -> getDoubleVar("dns","numerics"));
    std::cout << "In felix_driver: dew, dns = " << dew << "  " << dns << std::endl;
    dimInfoGeom = new int[dimInfo[0]+1];    
    for (int i=0;i<=dimInfo[0];i++) dimInfoGeom[i] = dimInfo[i];   
    std::cout << "DimInfoGeom  in felix_driver: " << std::endl;
    for (int i=0;i<=dimInfoGeom[0];i++) std::cout << dimInfoGeom[i] << " ";
    std::cout << std::endl;
    ewlb = *(btg_ptr -> getLongVar("ewlb","geometry"));
    ewub = *(btg_ptr -> getLongVar("ewub","geometry"));
    nslb = *(btg_ptr -> getLongVar("nslb","geometry"));
    nsub = *(btg_ptr -> getLongVar("nsub","geometry"));
    std::cout << "In felix_driver: ewlb, ewub = " << ewlb << "  " << ewub <<  std::endl;
    std::cout << "In felix_driver: nslb, nsub = " << nslb << "  " << nsub <<  std::endl;
    // define domain using dim_info
    ewn = dimInfoGeom[2];
    nsn = dimInfoGeom[3];


    // ---------------------------------------------
    // get constants from CISM
    // IK, 11/14/13: these things may not be needed in Albany/FELIX...  for now they are passed anyway.
    // ---------------------------------------------
 
    seconds_per_year_ptr = btg_ptr -> getDoubleVar("seconds_per_year","constants");
    gravity_ptr = btg_ptr -> getDoubleVar("gravity","constants");
    rho_ice_ptr = btg_ptr -> getDoubleVar("rho_ice","constants");
    rho_seawater_ptr = btg_ptr -> getDoubleVar("rho_seawater","constants");
    thicknessDataPtr = btg_ptr -> getDoubleVar("thck","geometry");
    topographyDataPtr = btg_ptr -> getDoubleVar("topg","geometry");
    upperSurfaceDataPtr = btg_ptr -> getDoubleVar("usrf","geometry");
    lowerSurfaceDataPtr = btg_ptr -> getDoubleVar("lsrf","geometry");
    floating_maskDataPtr = btg_ptr -> getDoubleVar("floating_mask","geometry");
    ice_maskDataPtr = btg_ptr -> getDoubleVar("ice_mask","geometry");
    lower_cell_locDataPtr = btg_ptr -> getDoubleVar("lower_cell_loc","geometry");

    // ---------------------------------------------
    // get connectivity arrays from CISM 
    // IK, 11/14/13: these things may not be needed in Albany/FELIX...  for now they are passed anyway.
    // ---------------------------------------------
    std::cout << "In felix_driver: grabbing connectivity array pointers from CISM..." << std::endl; 
    //IK, 11/13/13: check that connectivity derived types are transfered over from CISM to Albany/FELIX    
    nCellsActive = *(btg_ptr -> getLongVar("nCellsActive","connectivity")); 
    std::cout << "In felix_driver: nCellsActive = " << nCellsActive <<  std::endl;
    xyz_at_nodes_Ptr = btg_ptr -> getDoubleVar("xyz_at_nodes","connectivity"); 
    surf_height_at_nodes_Ptr = btg_ptr -> getDoubleVar("surf_height_at_nodes","connectivity"); 
    beta_at_nodes_Ptr = btg_ptr -> getDoubleVar("beta_at_nodes","connectivity");
    flwa_at_active_elements_Ptr = btg_ptr -> getDoubleVar("flwa_at_active_elements","connectivity"); 
    global_node_id_owned_map_Ptr = btg_ptr -> getLongVar("global_node_id_owned_map","connectivity");  
    global_element_conn_active_Ptr = btg_ptr -> getLongVar("global_element_conn_active","connectivity");  
    global_element_id_active_owned_map_Ptr = btg_ptr -> getLongVar("global_element_id_active_owned_map","connectivity");  
    global_basal_face_conn_active_Ptr = btg_ptr -> getLongVar("global_basal_face_conn_active","connectivity");  
    global_basal_face_id_active_owned_map_Ptr = btg_ptr -> getLongVar("global_basal_face_id_active_owned_map","connectivity");  
    global_basal_face_id_active_owned_map_Ptr = btg_ptr -> getLongVar("global_basal_face_id_active_owned_map","connectivity");  
    std::cout << "...done!" << std::endl; 

  
    // clean up
    std::cout << "exec mode = " << exec_mode << std::endl;

    std::cout << "End of nested scope." << std::endl; 
  }  

 

}


// updates cur_time_yr as solution is advanced
void felix_driver_run(FelixToGlimmer * btg_ptr, float& cur_time_yr, float time_inc_yr)
{
  Felix *felixPtr;
  
  std::cout << "In felix_driver_run, cur_time, time_inc = " 
       << cur_time_yr << "   " << time_inc_yr << std::endl;
 
  felix_store(btg_ptr -> getDyCoreIndex(), &felixPtr ,1);



}
  

void felix_driver_finalize(int amr_obj_index)
{
  Felix* felixPtr;

  std::cout << "In felix_driver_finalize..." << std::endl;

  felix_store(amr_obj_index, &felixPtr, 1);
  
  if (felixPtr != NULL)
    {
      //delete felixPtr; 
      felixPtr = NULL;
    }
  std::cout << "Felix Object deleted." << std::endl << std::endl; 
//#ifdef CH_MPI
  //  MPI_Finalize();
//#endif
  
    //return 0;
}

