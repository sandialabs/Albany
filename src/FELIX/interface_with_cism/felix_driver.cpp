
//uncomment the following if you want to write stuff out to matrix market to debug
//#define WRITE_TO_MATRIX_MARKET 

//uncomment the following if you want to exclude procs with 0 elements from solve.
//#define REDUCED_COMM

//uncomment the following if you want to use Epetra
//#define CISM_USE_EPETRA

//if we have an epetra build but no reduced comm, compute sensitivities 
//and responses
#ifdef CISM_USE_EPETRA
#ifndef REDUCED_COMM
#define COMPUTE_SENS_AND_RESP 
#endif
#endif

//if we have tpetra build, compute sensitivities and responses.
#ifndef CISM_USE_EPETRA
#define COMPUTE_SENS_AND_RESP
#endif

//computation of sensitivities and responses will be off in the case 
//we have an epetra build + reduced comm, as this was causing a hang.

#include <iostream>
#include <fstream>
#include "felix_driver.H"
#include "Albany_CismSTKMeshStruct.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Albany_Utils.hpp"
#include "Albany_SolverFactory.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include <stk_mesh/base/FieldBase.hpp>
#include "Piro_PerformSolve.hpp"
#include <stk_mesh/base/GetEntities.hpp>
#include "Albany_OrdinarySTKFieldContainer.hpp"
#ifdef CISM_USE_EPETRA
#include "Thyra_EpetraThyraWrappers.hpp"
#endif
//#include "Teuchos_TestForException.hpp"
#include "Kokkos_Core.hpp"

#ifdef WRITE_TO_MATRIX_MARKET
#ifdef CISM_USE_EPETRA
#include "EpetraExt_MultiVectorOut.h"
#include "EpetraExt_BlockMapOut.h"
#endif
#endif



Teuchos::RCP<Albany::CismSTKMeshStruct> meshStruct;
Teuchos::RCP<Albany::Application> albanyApp;
#ifdef CISM_USE_EPETRA
  Teuchos::RCP<const Epetra_Comm> mpiComm, reducedMpiComm;
  Teuchos::RCP<Thyra::ModelEvaluator<double> > solver;
#else
  Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<double> > solver;
#endif
Teuchos::RCP<const Teuchos_Comm> mpiCommT, reducedMpiCommT;
Teuchos::RCP<Teuchos::ParameterList> parameterList;
Teuchos::RCP<Teuchos::ParameterList> discParams;
Teuchos::RCP<Albany::SolverFactory> slvrfctry;
MPI_Comm comm, reducedComm;
bool interleavedOrdering; 
int nNodes2D; //number global nodes in the domain in 2D 
int nNodesProc2D; //number of nodes on each processor in 2D  
//vector used to renumber nodes on each processor from the Albany convention (horizontal levels first) to the CISM convention (vertical layers first)
std::vector<int> cismToAlbanyNodeNumberMap; 


int rank, number_procs;
long  cism_communicator; 
int cism_process_count, my_cism_rank;
double dew, dns;
long * dimInfo;        
int * dimInfoGeom; 
long ewlb, ewub, nslb, nsub;
long ewn, nsn, upn, nhalo; 
long global_ewn, global_nsn; 
double gravity, rho_ice, rho_seawater; //IK, 3/18/14: why are these pointers?  wouldn't they just be doubles? 
double final_time; //final time, added 10/30/14, IK 
double seconds_per_year, vel_scaling_param; 
//double * thicknessDataPtr, *topographyDataPtr;
//double * upperSurfaceDataPtr, * lowerSurfaceDataPtr;
//double * floating_maskDataPtr, * ice_maskDataPtr, * lower_cell_locDataPtr;
long nCellsActive;
long nWestFacesActive, nEastFacesActive, nSouthFacesActive, nNorthFacesActive; 
long debug_output_verbosity;
long use_glissade_surf_height_grad;
int nNodes, nElementsActive; 
//int nElementsActivePrevious = 0;  
double* xyz_at_nodes_Ptr, *surf_height_at_nodes_Ptr, *beta_at_nodes_Ptr, *thick_at_nodes_Ptr;
double* dsurf_height_at_nodes_dx_Ptr, *dsurf_height_at_nodes_dy_Ptr; 
double *flwa_at_active_elements_Ptr; 
int * global_node_id_owned_map_Ptr; 
int * global_element_conn_active_Ptr; 
int * global_element_id_active_owned_map_Ptr; 
int * global_basal_face_conn_active_Ptr; 
int * global_top_face_conn_active_Ptr; 
int * global_basal_face_id_active_owned_map_Ptr; 
int * global_top_face_id_active_owned_map_Ptr; 
int * global_west_face_conn_active_Ptr; 
int * global_west_face_id_active_owned_map_Ptr; 
int * global_east_face_conn_active_Ptr; 
int * global_east_face_id_active_owned_map_Ptr; 
int * global_south_face_conn_active_Ptr; 
int * global_south_face_id_active_owned_map_Ptr; 
int * global_north_face_conn_active_Ptr; 
int * global_north_face_id_active_owned_map_Ptr; 
int * dirichlet_node_mask_Ptr;
double *uVel_ptr; 
double *vVel_ptr; 
double *uvel_at_nodes_Ptr; 
double *vvel_at_nodes_Ptr; 
bool first_time_step = true;
#ifdef CISM_USE_EPETRA 
  Teuchos::RCP<Epetra_Map> node_map; 
#else
  Teuchos::RCP<Tpetra_Map> node_map; 
#endif
//Teuchos::RCP<Tpetra_Vector> previousSolution;
#ifdef CISM_USE_EPETRA
bool TpetraBuild = false; 
#else
bool TpetraBuild = true; 
#endif
bool keep_proc = true; 
const Tpetra::global_size_t INVALID = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid ();

#ifdef CISM_USE_EPETRA
Teuchos::RCP<const Epetra_Vector>
epetraVectorFromThyra(
  const Teuchos::RCP<const Epetra_Comm> &comm,
  const Teuchos::RCP<const Thyra::VectorBase<double> > &thyra)
{
  Teuchos::RCP<const Epetra_Vector> result;
  if (Teuchos::nonnull(thyra)) {
    const Teuchos::RCP<const Epetra_Map> epetra_map = Thyra::get_Epetra_Map(*thyra->space(), comm);
    result = Thyra::get_Epetra_Vector(*epetra_map, thyra);
  }
  return result;
}

Teuchos::RCP<const Epetra_MultiVector>
epetraMultiVectorFromThyra(
  const Teuchos::RCP<const Epetra_Comm> &comm,
  const Teuchos::RCP<const Thyra::MultiVectorBase<double> > &thyra)
{
  Teuchos::RCP<const Epetra_MultiVector> result;
  if (Teuchos::nonnull(thyra)) {
    const Teuchos::RCP<const Epetra_Map> epetra_map = Thyra::get_Epetra_Map(*thyra->range(), comm);
    result = Thyra::get_Epetra_MultiVector(*epetra_map, thyra);
  }
  return result;
}

void epetraFromThyra(
  const Teuchos::RCP<const Epetra_Comm> &comm,
  const Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<double> > > &thyraResponses,
  const Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Thyra::MultiVectorBase<double> > > > &thyraSensitivities,
  Teuchos::Array<Teuchos::RCP<const Epetra_Vector> > &responses,
  Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Epetra_MultiVector> > > &sensitivities)
{
  responses.clear();
  responses.reserve(thyraResponses.size());
  typedef Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<double> > > ThyraResponseArray;
  for (ThyraResponseArray::const_iterator it_begin = thyraResponses.begin(),
      it_end = thyraResponses.end(),
      it = it_begin;
      it != it_end;
      ++it) {
    responses.push_back(epetraVectorFromThyra(comm, *it));
  }

  sensitivities.clear();
  sensitivities.reserve(thyraSensitivities.size());
  typedef Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Thyra::MultiVectorBase<double> > > > ThyraSensitivityArray;
  for (ThyraSensitivityArray::const_iterator it_begin = thyraSensitivities.begin(),
      it_end = thyraSensitivities.end(),
      it = it_begin;
      it != it_end;
      ++it) {
    ThyraSensitivityArray::const_reference sens_thyra = *it;
    Teuchos::Array<Teuchos::RCP<const Epetra_MultiVector> > sens;
    sens.reserve(sens_thyra.size());
    for (ThyraSensitivityArray::value_type::const_iterator jt = sens_thyra.begin(),
        jt_end = sens_thyra.end();
        jt != jt_end;
        ++jt) {
        sens.push_back(epetraMultiVectorFromThyra(comm, *jt));
    }
    sensitivities.push_back(sens);
  }
}
#else
void tpetraFromThyra(
  const Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<ST> > > &thyraResponses,
  const Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Thyra::MultiVectorBase<ST> > > > &thyraSensitivities,
  Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> > &responses,
  Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Tpetra_MultiVector> > > &sensitivities)
{
  responses.clear();
  responses.reserve(thyraResponses.size());
  typedef Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<ST> > > ThyraResponseArray;
  for (ThyraResponseArray::const_iterator it_begin = thyraResponses.begin(),
      it_end = thyraResponses.end(),
      it = it_begin;
      it != it_end;
      ++it) {
    responses.push_back(Teuchos::nonnull(*it) ? ConverterT::getConstTpetraVector(*it) : Teuchos::null);
  }

  sensitivities.clear();
  sensitivities.reserve(thyraSensitivities.size());
  typedef Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Thyra::MultiVectorBase<ST> > > > ThyraSensitivityArray;
  for (ThyraSensitivityArray::const_iterator it_begin = thyraSensitivities.begin(),
      it_end = thyraSensitivities.end(),
      it = it_begin;
      it != it_end;
      ++it) {
  ThyraSensitivityArray::const_reference sens_thyra = *it;
    Teuchos::Array<Teuchos::RCP<const Tpetra_MultiVector> > sens;
    sens.reserve(sens_thyra.size());
    for (ThyraSensitivityArray::value_type::const_iterator jt = sens_thyra.begin(),
        jt_end = sens_thyra.end();
        jt != jt_end;
        ++jt) {
        sens.push_back(Teuchos::nonnull(*jt) ? ConverterT::getConstTpetraMultiVector(*jt) : Teuchos::null);
    }
    sensitivities.push_back(sens);
  }
}
#endif

void createReducedMPI(int nLocalEntities, MPI_Comm& reduced_comm_id) {
  int numProcs, me;
  MPI_Group world_group_id, reduced_group_id;
  MPI_Comm_size(comm, &numProcs);
  MPI_Comm_rank(comm, &me);
  std::vector<int> haveElements(numProcs);
  int nonEmpty = int(nLocalEntities > 0);
  MPI_Allgather(&nonEmpty, 1, MPI_INT, &haveElements[0], 1, MPI_INT, comm);
  std::vector<int> ranks;
  for (int i = 0; i < numProcs; i++) {
    if (haveElements[i])
      ranks.push_back(i);
  }

  MPI_Comm_group(comm, &world_group_id);
  MPI_Group_incl(world_group_id, ranks.size(), &ranks[0], &reduced_group_id);
  MPI_Comm_create(comm, reduced_group_id, &reduced_comm_id);
}


extern "C" void felix_driver_();

//What is exec_mode??
void felix_driver_init(int argc, int exec_mode, FelixToGlimmer * ftg_ptr, const char * input_fname)
{
   if (first_time_step) 
     Kokkos::initialize();  
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
    //MPI_COMM_size (comm, &cism_process_count); 
    //MPI_COMM_rank (comm, &my_cism_rank); 
    //convert comm to Epetra_Comm 
    //mpiComm = Albany::createEpetraCommFromMpiComm(reducedComm); 
#ifdef CISM_USE_EPETRA    
    mpiComm = Albany::createEpetraCommFromMpiComm(comm); 
#endif
    mpiCommT = Albany::createTeuchosCommFromMpiComm(comm); 
  
    //IK, 4/4/14: get verbosity level specified in CISM *.config file
    debug_output_verbosity = *(ftg_ptr -> getLongVar("debug_output_verbosity","options"));
    use_glissade_surf_height_grad = *(ftg_ptr -> getLongVar("use_glissade_surf_height_grad","options"));
    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0) 
      std::cout << "In felix_driver..." << std::endl;


    // ---------------------------------------------
    // get geometry info from CISM  
    //IK, 11/14/13: these things may not be needed in Albany/FELIX...  for now they are passed anyway.
    // ---------------------------------------------
    
    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0) 
      std::cout << "Getting geometry info from CISM..." << std::endl;
    dimInfo = ftg_ptr -> getLongVar("dimInfo","geometry");
    dew = *(ftg_ptr -> getDoubleVar("dew","numerics"));
    dns = *(ftg_ptr -> getDoubleVar("dns","numerics"));
    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0) 
      std::cout << "In felix_driver: dew, dns = " << dew << "  " << dns << std::endl;
    dimInfoGeom = new int[dimInfo[0]+1];    
    for (int i=0;i<=dimInfo[0];i++) dimInfoGeom[i] = dimInfo[i];   
    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0) {
      std::cout << "DimInfoGeom  in felix_driver: " << std::endl;
      for (int i=0;i<=dimInfoGeom[0];i++) std::cout << dimInfoGeom[i] << " ";
      std::cout << std::endl;
    }
    global_ewn = dimInfoGeom[2]; 
    global_nsn = dimInfoGeom[3]; 
    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0) {
       std::cout << "In felix_driver: global_ewn = " << global_ewn 
                 << ", global_nsn = " << global_nsn << std::endl;
    }
    ewlb = *(ftg_ptr -> getLongVar("ewlb","geometry"));
    ewub = *(ftg_ptr -> getLongVar("ewub","geometry"));
    nslb = *(ftg_ptr -> getLongVar("nslb","geometry"));
    nsub = *(ftg_ptr -> getLongVar("nsub","geometry"));
    nhalo = *(ftg_ptr -> getLongVar("nhalo","geometry"));
    ewn = *(ftg_ptr -> getLongVar("ewn","geometry"));
    nsn = *(ftg_ptr -> getLongVar("nsn","geometry"));
    upn = *(ftg_ptr -> getLongVar("upn","geometry"));
    if (debug_output_verbosity == 2) { 
      std::cout << "In felix_driver: Proc #" << mpiCommT->getRank() 
                << ", ewn = " << ewn << ", nsn = " << nsn << ", upn = " 
                << upn << ", nhalo = " << nhalo << std::endl;
    }

    // ---------------------------------------------
    // get constants from CISM
    // IK, 11/14/13: these things may not be needed in Albany/FELIX...  for now they are passed anyway.
    // ---------------------------------------------
 
    seconds_per_year = *(ftg_ptr -> getDoubleVar("seconds_per_year","constants"));
    vel_scaling_param = *(ftg_ptr -> getDoubleVar("vel_scaling_param","constants"));
    gravity = *(ftg_ptr -> getDoubleVar("gravity","constants"));
    rho_ice = *(ftg_ptr -> getDoubleVar("rho_ice","constants"));
    rho_seawater = *(ftg_ptr -> getDoubleVar("rho_seawater","constants"));
    //std::cout << "g, rho, rho_w: " << gravity << ", " << rho_ice << ", " << rho_seawater << std::endl; 
    final_time = *(ftg_ptr -> getDoubleVar("tend","numerics"));
    /*thicknessDataPtr = ftg_ptr -> getDoubleVar("thck","geometry");
    topographyDataPtr = ftg_ptr -> getDoubleVar("topg","geometry");
    upperSurfaceDataPtr = ftg_ptr -> getDoubleVar("usrf","geometry");
    lowerSurfaceDataPtr = ftg_ptr -> getDoubleVar("lsrf","geometry");
    floating_maskDataPtr = ftg_ptr -> getDoubleVar("floating_mask","geometry");
    ice_maskDataPtr = ftg_ptr -> getDoubleVar("ice_mask","geometry");
    lower_cell_locDataPtr = ftg_ptr -> getDoubleVar("lower_cell_loc","geometry");
    */

    // ---------------------------------------------
    // get connectivity arrays from CISM 
    // IK, 11/14/13: these things may not be needed in Albany/FELIX...  for now they are passed anyway.
    // ---------------------------------------------
    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0) 
      std::cout << "In felix_driver: grabbing connectivity array pointers from CISM..." << std::endl;
    //IK, 11/13/13: check that connectivity derived types are transfered over from CISM to Albany/FELIX    
    nCellsActive = *(ftg_ptr -> getLongVar("nCellsActive","connectivity"));
    nWestFacesActive = *(ftg_ptr -> getLongVar("nWestFacesActive","connectivity"));
    nEastFacesActive = *(ftg_ptr -> getLongVar("nEastFacesActive","connectivity"));
    nSouthFacesActive = *(ftg_ptr -> getLongVar("nSouthFacesActive","connectivity"));
    nNorthFacesActive = *(ftg_ptr -> getLongVar("nNorthFacesActive","connectivity"));
    if (debug_output_verbosity == 2) {  
      std::cout << "In felix_driver: Proc #" << mpiCommT->getRank() 
                << ", nCellsActive = " << nCellsActive 
                << ", nWestFacesActive = " << nWestFacesActive 
                << ", nEastFacesActive = " << nEastFacesActive 
                << ", nSouthFacesActive = " << nSouthFacesActive 
                << ", nNorthFacesActive = " << nNorthFacesActive <<std::endl;  
    }
    xyz_at_nodes_Ptr = ftg_ptr -> getDoubleVar("xyz_at_nodes","connectivity"); 
    surf_height_at_nodes_Ptr = ftg_ptr -> getDoubleVar("surf_height_at_nodes","geometry"); 
    thick_at_nodes_Ptr = ftg_ptr -> getDoubleVar("thick_at_nodes","geometry"); 
    dsurf_height_at_nodes_dx_Ptr = ftg_ptr -> getDoubleVar("dsurf_height_at_nodes_dx","geometry"); 
    dsurf_height_at_nodes_dy_Ptr = ftg_ptr -> getDoubleVar("dsurf_height_at_nodes_dy","geometry"); 
    beta_at_nodes_Ptr = ftg_ptr -> getDoubleVar("beta_at_nodes","velocity");
    flwa_at_active_elements_Ptr = ftg_ptr -> getDoubleVar("flwa_at_active_elements","temper"); 
    global_node_id_owned_map_Ptr = ftg_ptr -> getInt4Var("global_node_id_owned_map","connectivity");  
    global_element_conn_active_Ptr = ftg_ptr -> getInt4Var("global_element_conn_active","connectivity");  
    global_element_id_active_owned_map_Ptr = ftg_ptr -> getInt4Var("global_element_id_active_owned_map","connectivity");  
    global_basal_face_conn_active_Ptr = ftg_ptr -> getInt4Var("global_basal_face_conn_active","connectivity");  
    global_top_face_conn_active_Ptr = ftg_ptr -> getInt4Var("global_top_face_conn_active","connectivity");  
    global_basal_face_id_active_owned_map_Ptr = ftg_ptr -> getInt4Var("global_basal_face_id_active_owned_map","connectivity");  
    global_top_face_id_active_owned_map_Ptr = ftg_ptr -> getInt4Var("global_top_face_id_active_owned_map","connectivity");  
    global_west_face_conn_active_Ptr = ftg_ptr -> getInt4Var("global_west_face_conn_active","connectivity");  
    global_west_face_id_active_owned_map_Ptr = ftg_ptr -> getInt4Var("global_west_face_id_active_owned_map","connectivity");  
    global_east_face_conn_active_Ptr = ftg_ptr -> getInt4Var("global_east_face_conn_active","connectivity");  
    global_east_face_id_active_owned_map_Ptr = ftg_ptr -> getInt4Var("global_east_face_id_active_owned_map","connectivity");  
    global_south_face_conn_active_Ptr = ftg_ptr -> getInt4Var("global_south_face_conn_active","connectivity");  
    global_south_face_id_active_owned_map_Ptr = ftg_ptr -> getInt4Var("global_south_face_id_active_owned_map","connectivity");  
    global_north_face_conn_active_Ptr = ftg_ptr -> getInt4Var("global_north_face_conn_active","connectivity");  
    global_north_face_id_active_owned_map_Ptr = ftg_ptr -> getInt4Var("global_north_face_id_active_owned_map","connectivity");  
    dirichlet_node_mask_Ptr = ftg_ptr -> getInt4Var("dirichlet_node_mask","connectivity");  
   //get pointers to uvel and vvel from CISM for prescribing Dirichlet BC
    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0) 
      std::cout << "In felix_driver: grabbing pointers to u and v velocities in CISM..." << std::endl; 
    uvel_at_nodes_Ptr = ftg_ptr ->getDoubleVar("uvel_at_nodes", "velocity"); 
    vvel_at_nodes_Ptr = ftg_ptr ->getDoubleVar("vvel_at_nodes", "velocity"); 

//If requesting to do solve only on procs with > 0 elements, create reduced comm
#ifdef REDUCED_COMM
    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0) 
      std::cout << "In felix_driver: removing procs with 0 elements from computation (REDUCED_COMM set to ON)." << std::endl; 
    keep_proc = nCellsActive > 0;
    createReducedMPI(keep_proc, reducedComm);
#endif
    if (keep_proc) { //in the case we're using the reduced Comm, only call routines if there is a nonzero # of elts on a proc. 
#ifdef REDUCED_COMM 
      reducedMpiCommT = Albany::createTeuchosCommFromMpiComm(reducedComm);
   #ifdef CISM_USE_EPETRA
      reducedMpiComm = Albany::createEpetraCommFromMpiComm(reducedComm); 
   #endif
#else
      reducedMpiCommT = mpiCommT; 
   #ifdef CISM_USE_EPETRA
      reducedMpiComm = mpiComm; 
   #endif
#endif
    

    // ---------------------------------------------
    // create Albany mesh  
    // ---------------------------------------------
    // Read input file, the name of which is provided in the Glimmer/CISM .config file.
    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0) 
      std::cout << "In felix_driver: creating Albany mesh struct..." << std::endl;
    slvrfctry = Teuchos::rcp(new Albany::SolverFactory(input_fname, reducedMpiCommT));
    parameterList = Teuchos::rcp(&slvrfctry->getParameters(),false);
    discParams = Teuchos::sublist(parameterList, "Discretization", true);
    Albany::AbstractFieldContainer::FieldContainerRequirements req;
    int neq = 2; //number of equations - 2 for FO Stokes
    //IK, 11/14/13, debug output: check that pointers that are passed from CISM are not null 
    //std::cout << "DEBUG: xyz_at_nodes_Ptr: " << xyz_at_nodes_Ptr << std::endl; 
    //std::cout << "DEBUG: surf_height_at_nodes_Ptr: " << surf_height_at_nodes_Ptr << std::endl; 
    //std::cout << "DEBUG: thick_at_nodes_Ptr: " << thick_at_nodes_Ptr << std::endl; 
    //std::cout << "DEBUG: dsurf_height_at_nodes_dx_Ptr: " << dsurf_height_at_nodes_dx_Ptr << std::endl; 
    //std::cout << "DEBUG: dsurf_height_at_nodes_dy_Ptr: " << dsurf_height_at_nodes_dy_Ptr << std::endl; 
    //std::cout << "DEBUG: use_glissade_surf_height_grad: " << use_glissade_surf_height_grad << std::endl; 
    //std::cout << "DEBUG: beta_at_nodes_Ptr: " << beta_at_nodes_Ptr << std::endl; 
    //std::cout << "DEBUG: flwa_at_active_elements_Ptr: " << flwa_at_active_elements_Ptr << std::endl; 
    //std::cout << "DEBUG: global_node_id_owned_map_Ptr: " << global_node_id_owned_map_Ptr << std::endl; 
    //std::cout << "DEBUG: global_element_conn_active_Ptr: " << global_element_conn_active_Ptr << std::endl; 
    //std::cout << "DEBUG: global_basal_face_conn_active_Ptr: " << global_basal_face_conn_active_Ptr << std::endl; 
    //std::cout << "DEBUG: global_top_face_conn_active_Ptr: " << global_top_face_conn_active_Ptr << std::endl; 
    //std::cout << "DEBUG: global_basal_face_id_active_owned_map_Ptr: " << global_basal_face_id_active_owned_map_Ptr << std::endl;
    //std::cout << "DEBUG: global_top_face_id_active_owned_map_Ptr: " << global_top_face_id_active_owned_map_Ptr << std::endl;
    //std::cout << "DEBUG: global_west_face_conn_active_Ptr: " << global_west_face_conn_active_Ptr << std::endl; 
    //std::cout << "DEBUG: global_west_face_id_active_owned_map_Ptr: " << global_west_face_id_active_owned_map_Ptr << std::endl;
    //std::cout << "DEBUG: global_east_face_conn_active_Ptr: " << global_east_face_conn_active_Ptr << std::endl; 
    //std::cout << "DEBUG: global_east_face_id_active_owned_map_Ptr: " << global_east_face_id_active_owned_map_Ptr << std::endl;
    //std::cout << "DEBUG: global_south_face_conn_active_Ptr: " << global_south_face_conn_active_Ptr << std::endl; 
    //std::cout << "DEBUG: global_south_face_id_active_owned_map_Ptr: " << global_south_face_id_active_owned_map_Ptr << std::endl;
    //std::cout << "DEBUG: global_north_face_conn_active_Ptr: " << global_north_face_conn_active_Ptr << std::endl; 
    //std::cout << "DEBUG: global_north_face_id_active_owned_map_Ptr: " << global_north_face_id_active_owned_map_Ptr << std::endl;

    nNodes = (ewn-2*nhalo+1)*(nsn-2*nhalo+1)*upn; //number of nodes in mesh (on each processor) 
    nElementsActive = nCellsActive*(upn-1); //number of 3D active elements in mesh  
   
    //IK, 11/20/14: added this here: BCs are set in this file rather than in input.xml file to prevent confusion 
    //to the reader.  Basically the BCs should be passed from CISM.
    Teuchos::RCP<Teuchos::Array<double> >inputArrayBasal = Teuchos::rcp(new Teuchos::Array<double> (1, 1.0));
    parameterList->sublist("Problem").sublist("Neumann BCs").set("NBC on SS Basal for DOF all set basal_scalar_field", *inputArrayBasal);  
    //Lateral floating ice BCs. 
    if ((global_west_face_conn_active_Ptr != NULL || global_east_face_conn_active_Ptr != NULL || global_north_face_conn_active_Ptr != NULL || global_south_face_conn_active_Ptr != NULL) && (nWestFacesActive > 0 || nEastFacesActive > 0 || nSouthFacesActive > 0 || nNorthFacesActive > 0)) {
      Teuchos::RCP<Teuchos::Array<double> >inputArrayLateral = Teuchos::rcp(new Teuchos::Array<double> (1, rho_ice/rho_seawater));
      parameterList->sublist("Problem").sublist("Neumann BCs").set("NBC on SS Lateral for DOF all set lateral", *inputArrayLateral);  
    }
    //Dirichlet BCs
    if (dirichlet_node_mask_Ptr != NULL) {
      if ((uvel_at_nodes_Ptr != NULL) && (vvel_at_nodes_Ptr != NULL) ) {
        if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0) std::cout << "Setting Dirichlet BCs from CISM..." << std::endl;
        parameterList->sublist("Problem").sublist("Dirichlet BCs").set("DBC on NS NodeSetDirichlet for DOF U0 prescribe Field", "dirichlet_field");
        parameterList->sublist("Problem").sublist("Dirichlet BCs").set("DBC on NS NodeSetDirichlet for DOF U1 prescribe Field", "dirichlet_field");
        if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0) std::cout << "...done." << std::endl;
      }
      else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
          std::endl << "Error in felix_driver: cannot set Dirichlet BC from CISM; pointers to uvel and vvel passed from CISM are null."                    << std::endl);
      }
    }

 
   //IK, 11/20/14: pass gravity, ice density, and water density values to Albany.  These are needed 
   //in the PHAL_Neumann and FELIX_StokesFOBodyForce evaluators.  
    parameterList->sublist("Problem").sublist("FELIX Physical Parameters").set("Gravity", gravity);
    parameterList->sublist("Problem").sublist("FELIX Physical Parameters").set("Ice Density", rho_ice);
    parameterList->sublist("Problem").sublist("FELIX Physical Parameters").set("Water Density", rho_seawater);
 
    //IK, 11/17/14: if ds/dx, ds/dy are passed from CISM, use these in body force; 
    //otherwise calculate ds/dx from s by interpolation within Albany 
    if (dsurf_height_at_nodes_dx_Ptr != 0 && dsurf_height_at_nodes_dy_Ptr != 0) {
      parameterList->sublist("Problem").sublist("Body Force").set("Type", "FO Surface Grad Provided"); 
    }
    else {
      parameterList->sublist("Problem").sublist("Body Force").set("Type", "FO INTERP SURF GRAD"); 
    }

    albanyApp = Teuchos::rcp(new Albany::Application(reducedMpiCommT));
    albanyApp->initialSetUp(parameterList);
    meshStruct = Teuchos::rcp(new Albany::CismSTKMeshStruct(discParams, reducedMpiCommT, xyz_at_nodes_Ptr, global_node_id_owned_map_Ptr, 
                                                           global_element_id_active_owned_map_Ptr, 
                                                           global_element_conn_active_Ptr, 
                                                           global_basal_face_id_active_owned_map_Ptr, 
                                                           global_top_face_id_active_owned_map_Ptr, 
                                                           global_basal_face_conn_active_Ptr,
                                                           global_top_face_conn_active_Ptr,
                                                           global_west_face_id_active_owned_map_Ptr, 
                                                           global_west_face_conn_active_Ptr,
                                                           global_east_face_id_active_owned_map_Ptr, 
                                                           global_east_face_conn_active_Ptr,
                                                           global_south_face_id_active_owned_map_Ptr, 
                                                           global_south_face_conn_active_Ptr,
                                                           global_north_face_id_active_owned_map_Ptr, 
                                                           global_north_face_conn_active_Ptr,
                                                           dirichlet_node_mask_Ptr, 
                                                           uvel_at_nodes_Ptr, vvel_at_nodes_Ptr,  
                                                           beta_at_nodes_Ptr, surf_height_at_nodes_Ptr, 
                                                           dsurf_height_at_nodes_dx_Ptr, dsurf_height_at_nodes_dy_Ptr,
                                                           thick_at_nodes_Ptr, 
                                                           flwa_at_active_elements_Ptr, 
                                                           nNodes, nElementsActive, nCellsActive, 
                                                           nWestFacesActive, nEastFacesActive,
                                                           nSouthFacesActive, nNorthFacesActive,
                                                           debug_output_verbosity));

    albanyApp->createMeshSpecs(meshStruct);
    albanyApp->buildProblem();
    meshStruct->constructMesh(reducedMpiCommT, discParams, neq, req, albanyApp->getStateMgr().getStateInfoStruct(), meshStruct->getMeshSpecs()[0]->worksetSize);
 
    //Create node_map
    //global_node_id_owned_map_Ptr is 1-based, so node_map is 1-based
    //Distribute the elements according to the global element IDs
#ifdef CISM_USE_EPETRA
     node_map = Teuchos::rcp(new Epetra_Map(-1, nNodes, global_node_id_owned_map_Ptr, 0, *reducedMpiComm)); //node_map is 1-based
#else
    Teuchos::ArrayView<const GO> global_node_id_owned_map_AV = Teuchos::arrayView(global_node_id_owned_map_Ptr, nNodes);
    node_map = Teuchos::rcp(new Tpetra_Map(INVALID, global_node_id_owned_map_AV, 0, reducedMpiCommT));
#endif
 }


 
    // clean up
    //if (mpiComm->MyPID() == 0) std::cout << "exec mode = " << exec_mode << std::endl;
}

// The solve is done in the felix_driver_run function, and the solution is passed back to Glimmer-CISM 
// IK, 12/3/13: time_inc_yr and cur_time_yr are not used here... 
void felix_driver_run(FelixToGlimmer * ftg_ptr, double& cur_time_yr, double time_inc_yr)
{
    //IK, 12/9/13: how come FancyOStream prints an all processors??    
    Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0) {
      std::cout << "In felix_driver_run, cur_time, time_inc = " << cur_time_yr 
                << "   " << time_inc_yr << std::endl;
    }
    
    // ---------------------------------------------
    // get u and v velocity solution from Glimmer-CISM 
    // IK, 11/26/13: need to concatenate these into a single solve for initial condition for Albany/FELIX solve 
    // IK, 3/14/14: moved this step to felix_driver_run from felix_driver init, since we still want to grab and u and v velocities for CISM if the mesh hasn't changed, 
    // in which case only felix_driver_run will be called, not felix_driver_init.   
    // ---------------------------------------------
    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0) 
      std::cout << "In felix_driver_run: grabbing pointers to u and v velocities in CISM..." << std::endl; 
    uVel_ptr = ftg_ptr ->getDoubleVar("uvel", "velocity"); 
    vVel_ptr = ftg_ptr ->getDoubleVar("vvel", "velocity"); 

    // ---------------------------------------------
    // Set restart solution to the one passed from CISM
    // IK, 3/14/14: moved this from felix_driver_init to felix_driver_run.  
    // ---------------------------------------------
   
    if (keep_proc) { 
    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0) 
      std::cout << "In felix_driver_run: setting initial condition from CISM..." << std::endl;
    //Check what kind of ordering you have in the solution & create solutionField object.
    interleavedOrdering = meshStruct->getInterleavedOrdering();
    Albany::AbstractSTKFieldContainer::VectorFieldType* solutionField;
    if(interleavedOrdering)
      solutionField = Teuchos::rcp_dynamic_cast<Albany::OrdinarySTKFieldContainer<true> >(meshStruct->getFieldContainer())->getSolutionField();
    else
      solutionField = Teuchos::rcp_dynamic_cast<Albany::OrdinarySTKFieldContainer<false> >(meshStruct->getFieldContainer())->getSolutionField();

     //Create vector used to renumber nodes on each processor from the Albany convention (horizontal levels first) to the CISM convention (vertical layers first)
     nNodes2D = (global_ewn + 1)*(global_nsn+1); //number global nodes in the domain in 2D 
     nNodesProc2D = (nsn-2*nhalo+1)*(ewn-2*nhalo+1); //number of nodes on each processor in 2D  
     cismToAlbanyNodeNumberMap.resize(upn*nNodesProc2D);
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

     //The way it worked out, uVel_ptr and vVel_ptr have more nodes than the nodes in the mesh passed to Albany/CISM for the solve.  In particular, 
     //there is 1 row of halo elements in uVel_ptr and vVel_ptr.  To account for this, we copy uVel_ptr and vVel_ptr into std::vectors, which do not have the halo elements. 
     std::vector<double> uvel_vec(upn*nNodesProc2D); 
     std::vector<double> vvel_vec(upn*nNodesProc2D); 
     int counter1 = 0; 
     int counter2 = 0; 
     int local_nodeID; 
     for (int j=0; j<nsn-1; j++) {
       for (int i=0; i<ewn-1; i++) { 
         for (int k=0; k<upn; k++) {
           if (j >= nhalo-1 & j < nsn-nhalo) {
             if (i >= nhalo-1 & i < ewn-nhalo) {
#ifdef CISM_USE_EPETRA 
               local_nodeID = node_map->LID(cismToAlbanyNodeNumberMap[counter1]); 
#else
               local_nodeID = node_map->getLocalElement(cismToAlbanyNodeNumberMap[counter1]);
#endif
               uvel_vec[counter1] = uVel_ptr[counter2]; 
               vvel_vec[counter1] = vVel_ptr[counter2]; 
               counter1++;
            }
            }
            counter2++; 
         }
        }
     }
     //Loop over all the elements to find which nodes are active.  For the active nodes, copy uvel and vvel from CISM into Albany solution array to 
     //use as initial condition.
     //NOTE: there is some inefficiency here by looping over all the elements.  TO DO? pass only active nodes from Albany-CISM to improve this? 
     double velScale = seconds_per_year*vel_scaling_param;  
     for (int i=0; i<nElementsActive; i++) {
       for (int j=0; j<8; j++) {
        int node_GID =  global_element_conn_active_Ptr[i + nElementsActive*j]; //node_GID is 1-based
#ifdef CISM_USE_EPETRA      
        int node_LID =  node_map->LID(node_GID); //node_LID is 0-based
#else
        int node_LID =  node_map->getLocalElement(node_GID); //node_LID is 0-based
#endif
        stk::mesh::Entity node = meshStruct->bulkData->get_entity(stk::topology::NODE_RANK, node_GID);
        double* sol = stk::mesh::field_data(*solutionField, node);
        //IK, 3/18/14: added division by velScale to convert uvel and vvel from dimensionless to having units of m/year (the Albany units)  
        sol[0] = uvel_vec[node_LID]/velScale;
        sol[1] = vvel_vec[node_LID]/velScale;
      }
    }
    // ---------------------------------------------------------------------------------------------------
    // Solve 
    // ---------------------------------------------------------------------------------------------------

    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0) 
      std::cout << "In felix_driver_run: starting the solve... " << std::endl;
    //Need to set HasRestart solution such that uvel_Ptr and vvel_Ptr (u and v from Glimmer/CISM) are always set as initial condition?  
    meshStruct->setHasRestartSolution(!first_time_step);


    //Turn off homotopy if we're not in the first time-step. 
    //NOTE - IMPORTANT: Glen's Law Homotopy parameter should be set to 1.0 in the parameter list for this logic to work!!! 
    if (!first_time_step)
    {
       meshStruct->setRestartDataTime(parameterList->sublist("Problem").get("Homotopy Restart Step", 1.));
       double homotopy = parameterList->sublist("Problem").sublist("FELIX Viscosity").get("Glen's Law Homotopy Parameter", 1.0);
       if(meshStruct->restartDataTime()== homotopy) {
         parameterList->sublist("Problem").set("Solution Method", "Steady");
         parameterList->sublist("Piro").set("Solver Type", "NOX");
       }
    }

    albanyApp->createDiscretization();
    albanyApp->finalSetUp(parameterList); 

    //IK, 10/30/14: Check that # of elements from previous time step hasn't changed. 
    //If it has not, use previous solution as initial guess for current time step.
    //Otherwise do not set initial solution.  It's possible this can be improved so some part of the previous solution is used
    //defined on the current mesh (if it receded, which likely it will in dynamic ice sheet simulations...). 
    //if (nElementsActivePrevious != nElementsActive) previousSolution = Teuchos::null; 
    //albanyApp->finalSetUp(parameterList, previousSolution);

    //if (!first_time_step) 
    //  std::cout << "previousSolution: " << *previousSolution << std::endl;
#ifdef CISM_USE_EPETRA 
    solver = slvrfctry->createThyraSolverAndGetAlbanyApp(albanyApp, reducedMpiComm, reducedMpiComm, Teuchos::null, false);
#else
   solver = slvrfctry->createAndGetAlbanyAppT(albanyApp, reducedMpiCommT, reducedMpiCommT, Teuchos::null, false);
#endif

    Teuchos::ParameterList solveParams;
    solveParams.set("Compute Sensitivities", false);
    Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<double> > > thyraResponses;
    Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Thyra::MultiVectorBase<double> > > > thyraSensitivities;
    Piro::PerformSolveBase(*solver, solveParams, thyraResponses, thyraSensitivities);

#ifdef CISM_USE_EPETRA
    const Epetra_Map& ownedMap(*albanyApp->getDiscretization()->getMap()); //owned map
    const Epetra_Map& overlapMap(*albanyApp->getDiscretization()->getOverlapMap()); //overlap map
    Epetra_Import import(overlapMap, ownedMap); //importer from ownedMap to overlapMap
    Epetra_Vector solutionOverlap(overlapMap); //overlapped solution
    solutionOverlap.Import(*albanyApp->getDiscretization()->getSolutionField(), import, Insert);
#else 
    Teuchos::RCP<const Tpetra_Map> ownedMap = albanyApp->getDiscretization()->getMapT(); //owned map
    Teuchos::RCP<const Tpetra_Map> overlapMap = albanyApp->getDiscretization()->getOverlapMapT(); //overlap map
    Teuchos::RCP<Tpetra_Import> import = Teuchos::rcp(new Tpetra_Import(ownedMap, overlapMap));
    Teuchos::RCP<Tpetra_Vector> solutionOverlap = Teuchos::rcp(new Tpetra_Vector(overlapMap));
    solutionOverlap->doImport(*albanyApp->getDiscretization()->getSolutionFieldT(), *import, Tpetra::INSERT);
    Teuchos::ArrayRCP<const ST> solutionOverlap_constView = solutionOverlap->get1dView();
#endif

#ifdef WRITE_TO_MATRIX_MARKET
#ifdef CISM_USE_EPETRA
    //For debug: write solution and maps to matrix market file
    EpetraExt::BlockMapToMatrixMarketFile("node_map.mm", *node_map);
    EpetraExt::BlockMapToMatrixMarketFile("map.mm", ownedMap);
    EpetraExt::BlockMapToMatrixMarketFile("overlap_map.mm", overlapMap);
    EpetraExt::MultiVectorToMatrixMarketFile("solution.mm", *albanyApp->getDiscretization()->getSolutionField());
#else 
    Tpetra_MatrixMarket_Writer::writeMapFile("node_map.mm", *node_map);
    Tpetra_MatrixMarket_Writer::writeMapFile("map.mm", *ownedMap);
    Tpetra_MatrixMarket_Writer::writeMapFile("overlap_map.mm", *overlapMap);
    Tpetra_MatrixMarket_Writer::writeDenseFile("solution.mm", albanyApp->getDiscretization()->getSolutionFieldT());
#endif
#endif
   
   //set previousSolution (used as initial guess for next time step) to final Albany solution. 
   //previousSolution = Teuchos::rcp(new Tpetra_Vector(*albanyApp->getDiscretization()->getSolutionFieldT())); 
   //nElementsActivePrevious = nElementsActive;   
 
   //std::cout << "Final solution: " << *albanyApp->getDiscretization()->getSolutionField() << std::endl;  
    // ---------------------------------------------------------------------------------------------------
    // Compute sensitivies / responses and perform regression tests
    // IK, 12/9/13: how come this is turned off in mpas branch? 
    // ---------------------------------------------------------------------------------------------------

#ifdef COMPUTE_SENS_AND_RESP 
    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0) 
      std::cout << "Computing responses and sensitivities..." << std::endl;
    int status=0; // 0 = pass, failures are incremented
#ifdef CISM_USE_EPETRA
    Teuchos::Array<Teuchos::RCP<const Epetra_Vector> > responses;
    Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Epetra_MultiVector> > > sensitivities;
    epetraFromThyra(mpiComm, thyraResponses, thyraSensitivities, responses, sensitivities);
#else
    Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> > responses;
    Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Tpetra_MultiVector> > > sensitivities;
    tpetraFromThyra(thyraResponses, thyraSensitivities, responses, sensitivities);
#endif

    const int num_p = solver->Np(); // Number of *vectors* of parameters
    const int num_g = solver->Ng(); // Number of *vectors* of responses

   if (debug_output_verbosity != 0) {
    *out << "Finished eval of first model: Params, Responses "
      << std::setprecision(12) << std::endl;
   }
   const Thyra::ModelEvaluatorBase::InArgs<double> nominal = solver->getNominalValues();

   if (debug_output_verbosity != 0) {
    for (int i=0; i<num_p; i++) {
#ifdef CISM_USE_EPETRA
      const Teuchos::RCP<const Epetra_Vector> p_init = epetraVectorFromThyra(mpiComm, nominal.get_p(i));
      p_init->Print(*out << "\nParameter vector " << i << ":\n");
#else
      Albany::printTpetraVector(*out << "\nParameter vector " << i << ":\n",
           ConverterT::getConstTpetraVector(nominal.get_p(i)));
#endif
    }
   }

    for (int i=0; i<num_g-1; i++) {
#ifdef CISM_USE_EPETRA
      const Teuchos::RCP<const Epetra_Vector> g = responses[i];
#else
      const Teuchos::RCP<const Tpetra_Vector> g = responses[i];
#endif
      bool is_scalar = true;

      if (albanyApp != Teuchos::null)
        is_scalar = albanyApp->getResponse(i)->isScalarResponse();

      if (is_scalar) {
        if (debug_output_verbosity != 0) {
#ifdef CISM_USE_EPETRA
         g->Print(*out << "\nResponse vector " << i << ":\n");
#else
         Albany::printTpetraVector(*out << "\nResponse vector " << i << ":\n", g);
#endif
        }

        if (num_p == 0 && cur_time_yr == final_time) {
          // Just calculate regression data -- only if in final time step
#ifdef CISM_USE_EPETRA
          status += slvrfctry->checkSolveTestResults(i, 0, g.get(), NULL);
#else
          status += slvrfctry->checkSolveTestResultsT(i, 0, g.get(), NULL);
#endif
        } else {
          for (int j=0; j<num_p; j++) {
#ifdef CISM_USE_EPETRA
            const Teuchos::RCP<const Epetra_MultiVector> dgdp = sensitivities[i][j];
#else
            const Teuchos::RCP<const Tpetra_MultiVector> dgdp = sensitivities[i][j];
#endif
            if (debug_output_verbosity != 0) {
              if (Teuchos::nonnull(dgdp)) {
#ifdef CISM_USE_EPETRA
                dgdp->Print(*out << "\nSensitivities (" << i << "," << j << "):!\n");
#else
                Albany::printTpetraVector(*out << "\nSensitivities (" << i << "," << j << "):!\n", dgdp);
#endif
              }
            }
            if (cur_time_yr == final_time) {
#ifdef CISM_USE_EPETRA
              status += slvrfctry->checkSolveTestResults(i, j, g.get(), dgdp.get());
#else
              status += slvrfctry->checkSolveTestResultsT(i, j, g.get(), dgdp.get());
#endif
            }
          }
        }
      }
    }
    if (debug_output_verbosity != 0 && cur_time_yr == final_time) //only print regression test result if you're in the final time step 
      *out << "\nNumber of Failed Comparisons: " << status << std::endl;
    //IK, 10/30/14: added the following line so that when you run ctest from CISM the test fails if there are some failed comparisons.
    //if (status > 0)     
    //  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "All regression comparisons did not pass!" << std::endl);

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

    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0) 
      std::cout << "In felix_driver_run: copying Albany solution to uvel and vvel to send back to CISM... " << std::endl;
#ifdef CISM_USE_EPETRA 
    //Epetra_Vectors to hold uvel and vvel to be passed to Glimmer/CISM
    Epetra_Vector uvel(*node_map, true); 
    Epetra_Vector vvel(*node_map, true);
#else
    //Tpetra_Vectors to hold uvel and vvel to be passed to Glimmer/CISM
    Teuchos::RCP<Tpetra_Vector> uvel = Teuchos::rcp(new Tpetra_Vector(node_map, true));
    Teuchos::RCP<Tpetra_Vector> vvel = Teuchos::rcp(new Tpetra_Vector(node_map, true));
#endif

#ifdef CISM_USE_EPETRA 
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
#else
    if (interleavedOrdering == true) {
      for (int i=0; i<overlapMap->getNodeNumElements(); i++) {
        int global_dof = overlapMap->getGlobalElement(i);
        double sol_value = solutionOverlap_constView[i];
        int modulo = (global_dof % 2); //check if dof is for u or for v 
        int vel_global_dof, vel_local_dof;
        if (modulo == 0) { //u dof 
          vel_global_dof = global_dof/2+1; //add 1 because node_map is 1-based 
          vel_local_dof = node_map->getLocalElement(vel_global_dof); //look up local id corresponding to global id in node_map
          //std::cout << "uvel: global_dof = " << global_dof << ", uvel_global_dof = " << vel_global_dof << ", uvel_local_dof = " << vel_local_dof << std::endl; 
          uvel->replaceLocalValue(vel_local_dof, sol_value);
        }
        else { // v dof 
          vel_global_dof = (global_dof-1)/2+1; //add 1 because node_map is 1-based 
          vel_local_dof = node_map->getLocalElement(vel_global_dof); //look up local id corresponding to global id in node_map
          vvel->replaceLocalValue(vel_local_dof, sol_value);
        }
      }
    }
    else { //note: the case with non-interleaved ordering has not been tested...
      int numDofs = overlapMap->getGlobalNumElements();
      for (int i=0; i<overlapMap->getNodeNumElements(); i++) {
        int global_dof = overlapMap->getGlobalElement(i);
        double sol_value = solutionOverlap_constView[i];
        int vel_global_dof, vel_local_dof;
        if (global_dof < numDofs/2) { //u dof
          vel_global_dof = global_dof+1; //add 1 because node_map is 1-based 
          vel_local_dof = node_map->getLocalElement(vel_global_dof); //look up local id corresponding to global id in node_map
          uvel->replaceLocalValue(vel_local_dof, sol_value);
        }
        else { //v dofs 
          vel_global_dof = global_dof-numDofs/2+1; //add 1 because node_map is 1-based
          vel_local_dof = node_map->getLocalElement(vel_global_dof); //look up local id corresponding to global id in node_map
          vvel->replaceLocalValue(vel_local_dof, sol_value);
        }
      }
    }
#endif
 

#ifdef WRITE_TO_MATRIX_MARKET
    //For debug: write solution to matrix market file 
#ifdef CISM_USE_EPETRA
     EpetraExt::MultiVectorToMatrixMarketFile("uvel.mm", uvel); 
     EpetraExt::MultiVectorToMatrixMarketFile("vvel.mm", vvel);
#else
     Tpetra_MatrixMarket_Writer::writeDenseFile("uvel.mm", uvel);
     Tpetra_MatrixMarket_Writer::writeDenseFile("vvel.mm", vvel);
#endif
#endif
 
     //Copy uvel and vvel into uVel_ptr and vVel_ptr respectively (the arrays passed back to CISM) according to the numbering consistent w/ CISM. 
     counter1 = 0; 
     counter2 = 0;
#ifdef CISM_USE_EPETRA
#else
     Teuchos::ArrayRCP<const ST> uvel_constView = uvel->get1dView();
     Teuchos::ArrayRCP<const ST> vvel_constView = vvel->get1dView();
#endif 
     local_nodeID = 0;  
     for (int j=0; j<nsn-1; j++) {
       for (int i=0; i<ewn-1; i++) { 
         for (int k=0; k<upn; k++) {
           if (j >= nhalo-1 & j < nsn-nhalo) {
             if (i >= nhalo-1 & i < ewn-nhalo) {
#ifdef CISM_USE_EPETRA 
               local_nodeID = node_map->LID(cismToAlbanyNodeNumberMap[counter1]); 
               //if (mpiComm->MyPID() == 0) 
               //std::cout << "counter1:" << counter1 << ", cismToAlbanyNodeNumberMap[counter1]: " << cismToAlbanyNodeNumberMap[counter1] << ", local_nodeID: " 
               //<< local_nodeID << ", uvel: " << uvel[local_nodeID] << std::endl; //uvel[local_nodeID] << std::endl;  
               uVel_ptr[counter2] = uvel[local_nodeID];
               vVel_ptr[counter2] = vvel[local_nodeID];  
#else
               local_nodeID = node_map->getLocalElement(cismToAlbanyNodeNumberMap[counter1]);
               uVel_ptr[counter2] = uvel_constView[local_nodeID];
               vVel_ptr[counter2] = vvel_constView[local_nodeID];
#endif
               counter1++;
            }
            }
            else {
             uVel_ptr[counter2] = 0.0; 
             vVel_ptr[counter2] = 0.0; 
            }
            counter2++; 
         }
        }
      }
    }


    first_time_step = false;
    meshStruct = Teuchos::null;
    albanyApp = Teuchos::null;
    solver = Teuchos::null;
#ifdef CISM_USE_EPETRA
    mpiComm = Teuchos::null; 
    reducedMpiComm = Teuchos::null;
#endif
    if (cur_time_yr == final_time) 
      Kokkos::finalize(); 
}
  

//Clean up
//IK, 12/3/13: this is not called anywhere in the interface code...  used to be called (based on old bisicles interface code)?  
void felix_driver_finalize(int ftg_obj_index)
{
  if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0) {
    std::cout << "In felix_driver_finalize: cleaning up..." << std::endl;
    mpiCommT = Teuchos::null; 
    reducedMpiCommT = Teuchos::null;
    parameterList = Teuchos::null;
    discParams = Teuchos::null;
    slvrfctry = Teuchos::null;
    node_map = Teuchos::null; 
    
    //Should something happen here?? 
    std::cout << "done cleaning up!" << std::endl << std::endl; 
  }
}

