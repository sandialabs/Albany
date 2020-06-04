//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// Gather all Albany configuration macros
#include "Albany_config.h"

//uncomment the following if you want to write stuff out to matrix market to debug
//#define WRITE_TO_MATRIX_MARKET

//uncomment the following if you want to exclude procs with 0 elements from solve.
//#define REDUCED_COMM

//computation of sensitivities and responses will be off in the case
//we have an epetra build + reduced comm, as this was causing a hang.

#include "ali_driver.H"
#include "Albany_CismSTKMeshStruct.hpp"

#include "Albany_ThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_Utils.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_RegressionTests.hpp"
#include "Albany_OrdinarySTKFieldContainer.hpp"
#include "LandIce_ProblemFactory.hpp"

//#include "Teuchos_TestForException.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <Piro_PerformSolve.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <Kokkos_Core.hpp>
#include "Albany_GlobalLocalIndexer.hpp"

#include <iostream>
#include <fstream>

//FIXME: move static global variables to struct
//
//struct Global {...
//}
//
//static Global * g = nullptr; -- call in main
//
//delete g BEFORE calling Kokkos::finalize();
//

Teuchos::RCP<Albany::CismSTKMeshStruct> meshStruct;
Teuchos::RCP<Albany::Application> albanyApp;
Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<double> > solver;
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

Teuchos::RCP<const Thyra_VectorSpace> nodeVS;

//Teuchos::RCP<Tpetra_Vector> previousSolution;
bool keep_proc = true;
const GO INVALID = Teuchos::OrdinalTraits<GO>::invalid();

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


extern "C" void ali_driver_();

//What is exec_mode??
void ali_driver_init(int /* argc */, int /* exec_mode */, AliToGlimmer * ftg_ptr, const char * input_fname)
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
    mpiCommT = Albany::createTeuchosCommFromMpiComm(comm);

    //IK, 4/4/14: get verbosity level specified in CISM *.config file
    debug_output_verbosity = *(ftg_ptr -> getLongVar("debug_output_verbosity","options"));
    use_glissade_surf_height_grad = *(ftg_ptr -> getLongVar("use_glissade_surf_height_grad","options"));
    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0)
      std::cout << "In ali_driver..." << std::endl;


    // ---------------------------------------------
    // get geometry info from CISM
    //IK, 11/14/13: these things may not be needed in Albany/LandIce...  for now they are passed anyway.
    // ---------------------------------------------

    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0)
      std::cout << "Getting geometry info from CISM..." << std::endl;
    dimInfo = ftg_ptr -> getLongVar("dimInfo","geometry");
    dew = *(ftg_ptr -> getDoubleVar("dew","numerics"));
    dns = *(ftg_ptr -> getDoubleVar("dns","numerics"));
    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0)
      std::cout << "In ali_driver: dew, dns = " << dew << "  " << dns << std::endl;
    dimInfoGeom = new int[dimInfo[0]+1];
    for (int i=0;i<=dimInfo[0];i++) dimInfoGeom[i] = dimInfo[i];
    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0) {
      std::cout << "DimInfoGeom  in ali_driver: " << std::endl;
      for (int i=0;i<=dimInfoGeom[0];i++) std::cout << dimInfoGeom[i] << " ";
      std::cout << std::endl;
    }
    global_ewn = dimInfoGeom[2];
    global_nsn = dimInfoGeom[3];
    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0) {
       std::cout << "In ali_driver: global_ewn = " << global_ewn
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
      std::cout << "In ali_driver: Proc #" << mpiCommT->getRank()
                << ", ewn = " << ewn << ", nsn = " << nsn << ", upn = "
                << upn << ", nhalo = " << nhalo << std::endl;
    }

    // ---------------------------------------------
    // get constants from CISM
    // IK, 11/14/13: these things may not be needed in Albany/LandIce...  for now they are passed anyway.
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
    // IK, 11/14/13: these things may not be needed in Albany/LandIce...  for now they are passed anyway.
    // ---------------------------------------------
    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0)
      std::cout << "In ali_driver: grabbing connectivity array pointers from CISM..." << std::endl;
    //IK, 11/13/13: check that connectivity derived types are transfered over from CISM to Albany/LandIce
    nCellsActive = *(ftg_ptr -> getLongVar("nCellsActive","connectivity"));
    nWestFacesActive = *(ftg_ptr -> getLongVar("nWestFacesActive","connectivity"));
    nEastFacesActive = *(ftg_ptr -> getLongVar("nEastFacesActive","connectivity"));
    nSouthFacesActive = *(ftg_ptr -> getLongVar("nSouthFacesActive","connectivity"));
    nNorthFacesActive = *(ftg_ptr -> getLongVar("nNorthFacesActive","connectivity"));
    if (debug_output_verbosity == 2) {
      std::cout << "In ali_driver: Proc #" << mpiCommT->getRank()
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
      std::cout << "In ali_driver: grabbing pointers to u and v velocities in CISM..." << std::endl;
    uvel_at_nodes_Ptr = ftg_ptr ->getDoubleVar("uvel_at_nodes", "velocity");
    vvel_at_nodes_Ptr = ftg_ptr ->getDoubleVar("vvel_at_nodes", "velocity");

//If requesting to do solve only on procs with > 0 elements, create reduced comm
#ifdef REDUCED_COMM
    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0)
      std::cout << "In ali_driver: removing procs with 0 elements from computation (REDUCED_COMM set to ON)." << std::endl;
    keep_proc = nCellsActive > 0;
    createReducedMPI(keep_proc, reducedComm);
#endif
    if (keep_proc) { //in the case we're using the reduced Comm, only call routines if there is a nonzero # of elts on a proc.
#ifdef REDUCED_COMM
      reducedMpiCommT = Albany::createTeuchosCommFromMpiComm(reducedComm);
#else
      reducedMpiCommT = mpiCommT;
#endif



    // ---------------------------------------------
    // create Albany mesh
    // ---------------------------------------------
    // Read input file, the name of which is provided in the Glimmer/CISM .config file.
    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0)
      std::cout << "In ali_driver: creating Albany mesh struct..." << std::endl;
    slvrfctry = Teuchos::rcp(new Albany::SolverFactory(input_fname, reducedMpiCommT));
    const auto& bt = slvrfctry->getParameters()->get("Build Type","Tpetra");
    if (bt=="Tpetra") {
      // Set the static variable that denotes this as a Tpetra run
      static_cast<void>(Albany::build_type(Albany::BuildType::Tpetra));
//if we have tpetra build, compute sensitivities and responses.
#define COMPUTE_SENS_AND_RESP
    } 
    else if (bt=="Epetra") {
#ifndef ALBANY_EPETRA
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidArgument,
                                 "Error! You are attempting to run CALI with 'Build Type = Epetra', \n"
                                 "but Albany was configured with -DALBANY_EPETRA=OFF.  To run with this \n" 
                                 "build time, rebuild Albany with -DALBANY_EPETRA=ON.\n"); 
#endif
      // Set the static variable that denotes this as a Epetra run
      static_cast<void>(Albany::build_type(Albany::BuildType::Epetra));
      //if we have an epetra build but no reduced comm, compute sensitivities
      //and responses
#ifndef REDUCED_COMM
#define COMPUTE_SENS_AND_RESP
#endif
    } 
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidArgument,
                                 "Error! Invalid choice (" + bt + ") for 'BuildType'.\n"
                                 "       Valid choices are 'Epetra', 'Tpetra'.\n");
    }

    parameterList = slvrfctry->getParameters();
    discParams = Teuchos::sublist(parameterList, "Discretization", true);
    discParams->set<bool>("Output DTK Field to Exodus", true);
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

/*    std::string beta_name = "basal_friction";
    Teuchos::Array<std::string> arrayBasalFields(1, beta_name);
    Teuchos::Array<std::string> arraySideSets(1, "Basal");
    parameterList->sublist("Problem").set("Required Basal Fields", arrayBasalFields);
    parameterList->sublist("Problem").sublist("LandIce Basal Friction Coefficient").set<std::string>("Type", "Given Field");
    parameterList->sublist("Problem").set<std::string>("Basal Side Name",arraySideSets[0]);
    Teuchos::ParameterList& sideSetParamList = discParams->sublist("Side Set Discretizations");
    sideSetParamList.set("Side Sets", arraySideSets);
    Teuchos::ParameterList& basalParamList = sideSetParamList.sublist(arraySideSets[0]);
    basalParamList.set<int>("Number Of Time Derivatives",0);
    basalParamList.sublist("Required Fields Info").set<int>("Number Of Fields",1);
    basalParamList.sublist("Required Fields Info").sublist("Field 0").set<std::string>("Field Name",beta_name);
    basalParamList.sublist("Required Fields Info").sublist("Field 0").set<std::string>("Field Type","From Mesh");*/

    Teuchos::Array<std::string> arrayRequiredFields(8);
    arrayRequiredFields[0]="flow_factor"; arrayRequiredFields[1]="temperature";
    arrayRequiredFields[2]="ice_thickness"; arrayRequiredFields[3]="surface_height";
    arrayRequiredFields[4]="basal_friction"; arrayRequiredFields[5]="dirichlet_field";
    arrayRequiredFields[6]="xgrad_surface_height"; arrayRequiredFields[7]="ygrad_surface_height";

    parameterList->sublist("Problem").set("Required Fields", arrayRequiredFields);

    // --- LandIce-specific boundary conditions --- //
    int numLandIceBCs;
    if ((global_west_face_conn_active_Ptr != NULL || global_east_face_conn_active_Ptr != NULL || global_north_face_conn_active_Ptr != NULL || global_south_face_conn_active_Ptr != NULL) && (nWestFacesActive > 0 || nEastFacesActive > 0 || nSouthFacesActive > 0 || nNorthFacesActive > 0)) {
      numLandIceBCs = 2;
    } else {
      numLandIceBCs = 1;
    }

    parameterList->sublist("Problem").sublist("LandIce BCs").set("Number",numLandIceBCs);

    // Basal friction sliding bc
    parameterList->sublist("Problem").sublist("LandIce BCs").sublist("BC 0").set<std::string>("Type","Basal Friction");
    parameterList->sublist("Problem").sublist("LandIce BCs").sublist("BC 0").set<std::string>("Side Set Name","Basal");
    parameterList->sublist("Problem").sublist("LandIce BCs").sublist("BC 0").set<int>("Cubature Degree",3);
    parameterList->sublist("Problem").sublist("LandIce BCs").sublist("BC 0").sublist("Basal Friction Coefficient").set<std::string>("Type","Given Field");
    parameterList->sublist("Problem").sublist("LandIce BCs").sublist("BC 0").sublist("Basal Friction Coefficient").set<std::string>("Given Field Variable Name", "basal_friction");

    //Lateral floating ice BCs.
    if ((global_west_face_conn_active_Ptr != NULL || global_east_face_conn_active_Ptr != NULL || global_north_face_conn_active_Ptr != NULL || global_south_face_conn_active_Ptr != NULL) && (nWestFacesActive > 0 || nEastFacesActive > 0 || nSouthFacesActive > 0 || nNorthFacesActive > 0)) {
      parameterList->sublist("Problem").sublist("LandIce BCs").sublist("BC 1").set<std::string>("Type","Lateral");
      parameterList->sublist("Problem").sublist("LandIce BCs").sublist("BC 1").set<std::string>("Side Set Name","Lateral");
      parameterList->sublist("Problem").sublist("LandIce BCs").sublist("BC 1").set<int>("Cubature Degree",3);
      parameterList->sublist("Problem").sublist("LandIce BCs").sublist("BC 1").set<double>("Immersed Ratio",rho_ice/rho_seawater);
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
          std::endl << "Error in ali_driver: cannot set Dirichlet BC from CISM; pointers to uvel and vvel passed from CISM are null."                    << std::endl);
      }
    }

   //IK, 11/20/14: pass gravity, ice density, and water density values to Albany.  These are needed
   //in the PHAL_Neumann and LandIce_StokesFOBodyForce evaluators.
    parameterList->sublist("Problem").sublist("LandIce Physical Parameters").set("Gravity Acceleration", gravity);
    parameterList->sublist("Problem").sublist("LandIce Physical Parameters").set("Ice Density", rho_ice);
    parameterList->sublist("Problem").sublist("LandIce Physical Parameters").set("Water Density", rho_seawater);

    //IK, 11/17/14: if ds/dx, ds/dy are passed from CISM, use these in body force;
    //otherwise calculate ds/dx from s by interpolation within Albany
    if (dsurf_height_at_nodes_dx_Ptr != 0 && dsurf_height_at_nodes_dy_Ptr != 0) {
      parameterList->sublist("Problem").sublist("Body Force").set("Type", "FO Surface Grad Provided");
    }
    else {
      parameterList->sublist("Problem").sublist("Body Force").set("Type", "FO INTERP SURF GRAD");
    }


    discParams->set<std::string>("Method", "Cism");
    discParams->sublist("Required Fields Info").set<int>("Number Of Fields",8);
    Teuchos::ParameterList& field0 = discParams->sublist("Required Fields Info").sublist("Field 0");
    Teuchos::ParameterList& field1 = discParams->sublist("Required Fields Info").sublist("Field 1");
    Teuchos::ParameterList& field2 = discParams->sublist("Required Fields Info").sublist("Field 2");
    Teuchos::ParameterList& field3 = discParams->sublist("Required Fields Info").sublist("Field 3");
    Teuchos::ParameterList& field4 = discParams->sublist("Required Fields Info").sublist("Field 4");
    Teuchos::ParameterList& field5 = discParams->sublist("Required Fields Info").sublist("Field 5");
    Teuchos::ParameterList& field6 = discParams->sublist("Required Fields Info").sublist("Field 6");
    Teuchos::ParameterList& field7 = discParams->sublist("Required Fields Info").sublist("Field 7");

    //set flow_factor
    field0.set<std::string>("Field Name", "flow_factor");
    field0.set<std::string>("Field Type", "Elem Scalar");
    field0.set<std::string>("Field Origin", "Mesh");

    //set temperature
    field1.set<std::string>("Field Name", "temperature");
    field1.set<std::string>("Field Type", "Elem Scalar");
    field1.set<std::string>("Field Origin", "Mesh");

    //set ice thickness
    field2.set<std::string>("Field Name", "ice_thickness");
    field2.set<std::string>("Field Type", "Node Scalar");
    field2.set<std::string>("Field Origin", "Mesh");

    //set ice surface height
    field3.set<std::string>("Field Name", "surface_height");
    field3.set<std::string>("Field Type", "Node Scalar");
    field3.set<std::string>("Field Origin", "Mesh");

    //set basal_friction
    field4.set<std::string>("Field Name", "basal_friction");
    field4.set<std::string>("Field Type", "Node Scalar");
    field4.set<std::string>("Field Origin", "Mesh");

    //set dirichlet_field
    field5.set<std::string>("Field Name", "dirichlet_field");
    field5.set<std::string>("Field Type", "Node Vector");
    field5.set<std::string>("Field Origin", "Mesh");

    //set ice xgrad surface height
    field6.set<std::string>("Field Name", "xgrad_surface_height");
    field6.set<std::string>("Field Type", "Node Scalar");
    field6.set<std::string>("Field Origin", "Mesh");

    //set ice ygrad surface height
    field7.set<std::string>("Field Name", "ygrad_surface_height");
    field7.set<std::string>("Field Type", "Node Scalar");
    field7.set<std::string>("Field Origin", "Mesh");

    // Register LandIce problems
    auto& pb_factories = Albany::FactoriesContainer<Albany::ProblemFactory>::instance();
    pb_factories.add_factory(LandIce::LandIceProblemFactory::instance());

    albanyApp = Teuchos::rcp(new Albany::Application(reducedMpiCommT));
    albanyApp->initialSetUp(parameterList);

    //Get number of params in problem - needed for MeshStruct constructor
    int num_params = Albany::CalculateNumberParams(Teuchos::sublist(parameterList, "Problem", true)); 
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
                                                           num_params, debug_output_verbosity));

    albanyApp->createMeshSpecs(meshStruct);
    albanyApp->buildProblem();
    meshStruct->constructMesh(reducedMpiCommT, discParams, neq, req, albanyApp->getStateMgr().getStateInfoStruct(), meshStruct->getMeshSpecs()[0]->worksetSize);

    //Create nodeVS
    //global_node_id_owned_map_Ptr is 1-based, so nodeVS is 1-based
    //Distribute the elements according to the global element IDs
    Teuchos::Array<GO> global_node_id_owned_map(nNodes);
    for (int i=0; i<nNodes; i++) {
      global_node_id_owned_map[i] = global_node_id_owned_map_Ptr[i];
    }
    nodeVS = Albany::createVectorSpace(reducedMpiCommT, global_node_id_owned_map(), INVALID);  
 }



    // clean up
    //if (mpiComm->MyPID() == 0) std::cout << "exec mode = " << exec_mode << std::endl;
}

// The solve is done in the ali_driver_run function, and the solution is passed back to Glimmer-CISM
// IK, 12/3/13: time_inc_yr and cur_time_yr are not used here...
void ali_driver_run(AliToGlimmer * ftg_ptr, double& cur_time_yr, double time_inc_yr)
{
    //IK, 12/9/13: how come FancyOStream prints an all processors??
    Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0) {
      std::cout << "In ali_driver_run, cur_time, time_inc = " << cur_time_yr
                << "   " << time_inc_yr << std::endl;
    }

    // ---------------------------------------------
    // get u and v velocity solution from Glimmer-CISM
    // IK, 11/26/13: need to concatenate these into a single solve for initial condition for Albany/LandIce solve
    // IK, 3/14/14: moved this step to ali_driver_run from ali_driver init, since we still want to grab and u and v velocities for CISM if the mesh hasn't changed,
    // in which case only ali_driver_run will be called, not ali_driver_init.
    // ---------------------------------------------
    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0)
      std::cout << "In ali_driver_run: grabbing pointers to u and v velocities in CISM..." << std::endl;
    uVel_ptr = ftg_ptr ->getDoubleVar("uvel", "velocity");
    vVel_ptr = ftg_ptr ->getDoubleVar("vvel", "velocity");

    // ---------------------------------------------
    // Set restart solution to the one passed from CISM
    // IK, 3/14/14: moved this from ali_driver_init to ali_driver_run.
    // ---------------------------------------------

    if (keep_proc) {
    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0)
      std::cout << "In ali_driver_run: setting initial condition from CISM..." << std::endl;
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
     auto indexer = Albany::createGlobalLocalIndexer(nodeVS);
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
               local_nodeID = indexer->getLocalElement(cismToAlbanyNodeNumberMap[counter1]);
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
        auto node_LID = indexer->getLocalElement(node_GID);
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
      std::cout << "In ali_driver_run: starting the solve... " << std::endl;
    //Need to set HasRestart solution such that uvel_Ptr and vvel_Ptr (u and v from Glimmer/CISM) are always set as initial condition?
    meshStruct->setHasRestartSolution(!first_time_step);


    //Turn off homotopy if we're not in the first time-step.
    //NOTE - IMPORTANT: Glen's Law Homotopy parameter should be set to 1.0 in the parameter list for this logic to work!!!
    if (!first_time_step)
    {
       meshStruct->setRestartDataTime(parameterList->sublist("Problem").get("Homotopy Restart Step", 1.));
       double homotopy = parameterList->sublist("Problem").sublist("LandIce Viscosity").get("Glen's Law Homotopy Parameter", 1.0);
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
    auto albanyModel = slvrfctry->createModel(albanyApp);
    solver = slvrfctry->createSolver(albanyModel, reducedMpiCommT);

    Teuchos::ParameterList solveParams;
    solveParams.set("Compute Sensitivities", false);
    Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<double> > > thyraResponses;
    Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Thyra::MultiVectorBase<double> > > > thyraSensitivities;
    Piro::PerformSolveBase(*solver, solveParams, thyraResponses, thyraSensitivities);

    auto disc = albanyApp->getDiscretization();
    auto ownedVS = disc->getVectorSpace();
    auto overlapVS = disc->getOverlapVectorSpace();
    auto cas_manager = Albany::createCombineAndScatterManager(ownedVS,overlapVS);
    auto solutionOverlap = Thyra::createMember(overlapVS);
    cas_manager->scatter(*disc->getSolutionField(),*solutionOverlap,Albany::CombineMode::INSERT);
    auto solutionOverlap_constView = Albany::getLocalData(Teuchos::rcp_dynamic_cast<const Thyra_Vector>(solutionOverlap)); 

#ifdef WRITE_TO_MATRIX_MARKET
    //For debug: write solution and maps to matrix market file
    Albany::writeMatrixMarket(nodeVS, "nodeVS");  
    Albany::writeMatrixMarket(ownedVS, "ownedVs");  
    Albany::writeMatrixMarket(overlapVS, "overlapVs");  
    Albany::writeMatrixMarket(albanyApp->getDiscretization()->getSolutionField(), "solution");  
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

    const int num_p = solver->Np(); // Number of *vectors* of parameters
    const int num_g = solver->Ng(); // Number of *vectors* of responses

   if (debug_output_verbosity != 0) {
    *out << "Finished eval of first model: Params, Responses "
      << std::setprecision(12) << std::endl;
   }
   const Thyra::ModelEvaluatorBase::InArgs<double> nominal = solver->getNominalValues();

   if (debug_output_verbosity != 0) {
    for (int i=0; i<num_p; i++) {
      Albany::printThyraVector(*out << "\nParameter vector " << i << ":\n", nominal.get_p(i));
    }
   }

    Albany::RegressionTests regression(slvrfctry->getParameters());
    for (int i=0; i<num_g-1; i++) {
      const Teuchos::RCP<const Thyra_Vector> g = thyraResponses[i];

      bool is_scalar = true;

      if (albanyApp != Teuchos::null)
        is_scalar = albanyApp->getResponse(i)->isScalarResponse();

      if (is_scalar) {
        if (debug_output_verbosity != 0) {
          Albany::printThyraVector(*out << "\nResponse vector " << i << ":\n", g);
        }

        if (num_p == 0 && cur_time_yr == final_time) {
          // Just calculate regression data -- only if in final time step
          status += regression.checkSolveTestResults(i, 0, g, Teuchos::null);
        } else {
          for (int j=0; j<num_p; j++) {
            Teuchos::RCP<const Thyra_MultiVector> dgdp = thyraSensitivities[i][j];

            if (debug_output_verbosity != 0) {
              if (Teuchos::nonnull(dgdp)) {
                Albany::printThyraMultiVector(*out <<"\nSensitivities (" << i << ", " << j << "):\n", dgdp); 
              }
            }
            if (cur_time_yr == final_time) {
              status += regression.checkSolveTestResults(i, j, g, dgdp);
            }
          }
        }
      }
    }
    if (debug_output_verbosity != 0 && cur_time_yr == final_time) //only print regression test result if you're in the final time step
      *out << "\nNumber of Failed Comparisons: " << status << std::endl;
#ifdef CISM_CHECK_COMPARISONS
    //IK, 10/30/14: added the following line so that when you run ctest from CISM the test fails if there are some failed comparisons.
    if (status > 0)
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "All regression comparisons did not pass!" << std::endl);
#endif

#endif
    // ---------------------------------------------------------------------------------------------------
    // Copy solution back to glimmer uvel and vvel arrays to be passed back
    // ---------------------------------------------------------------------------------------------------

    //std::cout << "overlapMap # global elements: " << overlapMap.NumGlobalElements() << std::endl;
    //std::cout << "overlapMap # my elements: " << overlapMap.NumMyElements() << std::endl;
    //std::cout << "overlapMap: " << overlapMap << std::endl;
    //std::cout << "map # global elements: " << ownedMap.NumGlobalElements() << std::endl;
    //std::cout << "map # my elements: " << ownedMap.NumMyElements() << std::endl;
    //std::cout << "nodeVS # global elements: " << nodeVS->NumGlobalElements() << std::endl;
    //std::cout << "nodeVS # my elements: " << nodeVS->NumMyElements() << std::endl;
    //std::cout << "nodeVS: " << *nodeVS << std::endl;

    if (debug_output_verbosity != 0 & mpiCommT->getRank() == 0)
      std::cout << "In ali_driver_run: copying Albany solution to uvel and vvel to send back to CISM... " << std::endl;

    //IKT, 10/6/17: the following is towards fixing Albany github issue #187, having to do
    //with the solution passed to the *.nc file not being copied from Albany to CISM
    //correctly in parallel for all geometries/decompositions.

    int numGlobalNodes;
    numGlobalNodes = nodeVS->dim(); 

    //IKT, 10/6/17: ensure size of *vel_local* and *vec_global* std::vecs
    //is large enough when reduced comm is used.
#ifdef REDUCED_COMM
    numGlobalNodes*=2;
#endif

    std::vector<double> uvel_local_vec(numGlobalNodes, 0.0);
    std::vector<double> vvel_local_vec(numGlobalNodes, 0.0);

    int overlap_vs_num_my_elts;
    int global_dof;
    double sol_value;
    int numDofs;
    auto ov_vs_indexer = Albany::createGlobalLocalIndexer(overlapVS);
    overlap_vs_num_my_elts = ov_vs_indexer->getNumLocalElements(); 

    if (interleavedOrdering == true) {
      for (int i=0; i<overlap_vs_num_my_elts; i++) {
        global_dof = ov_vs_indexer->getGlobalElement(i); 
        sol_value = solutionOverlap_constView[i]; 
        int modulo = (global_dof % 2); //check if dof is for u or for v
        int vel_global_dof;
        if (modulo == 0) { //u dof
          vel_global_dof = global_dof/2+1; //add 1 because nodeVS is 1-based
          uvel_local_vec[vel_global_dof] = sol_value;
        }
        else { // v dof
          vel_global_dof = (global_dof-1)/2+1; //add 1 because nodeVS is 1-based
          vvel_local_vec[vel_global_dof] = sol_value;
        }
      }
    }
    else { //note: the case with non-interleaved ordering has not been tested...
      numDofs = ov_vs_indexer->getNumLocalElements(); 
      for (int i=0; i<overlap_vs_num_my_elts; i++) {
        global_dof = ov_vs_indexer->getGlobalElement(i); 
        sol_value = solutionOverlap_constView[i];
        int vel_global_dof;
        if (global_dof < numDofs/2) { //u dof
          vel_global_dof = global_dof+1; //add 1 because nodeVS is 1-based
          uvel_local_vec[vel_global_dof] = sol_value;
        }
        else { //v dofs
          vel_global_dof = global_dof-numDofs/2+1; //add 1 because nodeVS is 1-based
          vvel_local_vec[vel_global_dof] = sol_value;
        }
      }
    }

     //Copy uvel and vvel into uVel_ptr and vVel_ptr respectively (the arrays passed back to CISM) according to the numbering consistent w/ CISM.
     counter1 = 0;
     counter2 = 0;
     for (int j=0; j<nsn-1; j++) {
       for (int i=0; i<ewn-1; i++) {
         for (int k=0; k<upn; k++) {
           if (j >= nhalo-1 & j < nsn-nhalo) {
             if (i >= nhalo-1 & i < ewn-nhalo) {
               auto global_nodeID = cismToAlbanyNodeNumberMap[counter1];
               uVel_ptr[counter2] = uvel_local_vec[global_nodeID];
               vVel_ptr[counter2] = vvel_local_vec[global_nodeID];
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
    if (cur_time_yr == final_time) {
      mpiCommT = Teuchos::null;
      reducedMpiCommT = Teuchos::null;
      parameterList = Teuchos::null;
      discParams = Teuchos::null;
      slvrfctry = Teuchos::null;
      nodeVS = Teuchos::null;
      Kokkos::finalize();
    }
}

//Clean up
//IK, 12/3/13: this is not called anywhere in the interface code...  used to be called (based on old bisicles interface code)?
void ali_driver_finalize(int ftg_obj_index)
{
  if (debug_output_verbosity != 0 && mpiCommT->getRank() == 0) {
    std::cout << "In ali_driver_finalize: cleaning up..." << std::endl;
  }

  //Nothing to do.

  if (debug_output_verbosity != 0 && mpiCommT->getRank() == 0) {
    std::cout << "...done cleaning up!" << std::endl << std::endl;
  }
}
