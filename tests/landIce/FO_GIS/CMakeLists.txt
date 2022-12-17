# Name the test with the directory name

get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${parentDirName}_${dirName})

# This variable will be set in the input files during the 'configure_file' call
if (ALBANY_PARALELL_EXODUS)
  set (USE_SERIAL_MESH "true")
#IKT 5/22/19 - commenting out for now as this is leading to compilation error
else ()
  # If there's no Iopx, we *must* have ALBANY_DECOMP defined (to decompose the mesh)
  # OR execute with only one rank.
  IF (ALBANY_MPI AND NOT ${ALBANY_SEACAS_PATH})
    MESSAGE (FATAL_ERROR "Error! Cannot execute FO_GIS tests in parallel without Iopx or decomp from Trilinos.\n")
  ENDIF()
 set (USE_SERIAL_MESH "false")
endif()


###########################################
###     Import 2d ascii mesh (test)     ###
###########################################

set (testName ${testNameRoot}_Import2DAsciiMesh)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_create_exo_from_msh.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_create_exo_from_msh.yaml)
 
IF (ALBANY_PARALELL_EXODUS AND ALBANY_MPI)
  add_test(${testName} ${Albany.exe} input_create_exo_from_msh.yaml)  
  set_tests_properties(${testName}
                       PROPERTIES
                       LABELS         "LandIce")  
ENDIF ()

add_test(${testName}_Serial ${SerialAlbany.exe} input_create_exo_from_msh.yaml)
          
set_tests_properties(${testName}_Serial
                       PROPERTIES
                       LABELS         "LandIce")


####################################
###     Populate mesh (test)     ###
####################################

set (testName ${testNameRoot}_PopulateMeshes)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_populate_meshes.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_populate_meshes.yaml)

add_test(${testName} ${Albany.exe} input_fo_gis_populate_meshes.yaml)
# If Iopx is not present, we must decompose the input mesh first
IF (NOT ALBANY_PARALELL_EXODUS AND ALBANY_MPI)
  set (testNameDecompMesh ${testNameRoot}_decompMesh)
  add_test (NAME ${testNameDecompMesh}
            COMMAND ${SEACAS_DECOMP} -processors ${MPIMNP} gis_unstruct_2d.exo
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../ExoMeshes)
  set_tests_properties (${testNameDecompMesh} PROPERTIES FIXTURES_SETUP meshSetup RUN_SERIAL TRUE)

  set_tests_properties(${testName}
                       PROPERTIES
                       LABELS            "LandIce;Epetra;Forward"
                       FIXTURES_SETUP    PopulateMeshes
                       FIXTURES_REQUIRED meshSetup)
ELSE ()
  set_tests_properties(${testName}
                       PROPERTIES
                       LABELS         "LandIce;Epetra;Forward"
                       FIXTURES_SETUP PopulateMeshes)
ENDIF()


####################################
###        Unstruct tests        ###
####################################

set (testName ${testNameRoot}_Unstructured)

if (ALBANY_EPETRA)
  # First run
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_unstruct.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_unstruct.yaml)
  add_test(${testName}_Epetra ${Albany.exe} input_fo_gis_unstruct.yaml)
  set_tests_properties(${testName}_Epetra
                       PROPERTIES
                       LABELS            "LandIce;Epetra;Forward"
                       FIXTURES_REQUIRED PopulateMeshes
                       FIXTURES_SETUP    UnstructRestart)

  # Restart
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_unstruct_restart.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_unstruct_restart.yaml)
  add_test(${testName}_Restart ${Albany.exe} input_fo_gis_unstruct_restart.yaml)
  set_tests_properties(${testName}_Restart
                       PROPERTIES
                       LABELS "LandIce;Epetra;Forward"
                       RUN_SERIAL TRUE
                       FIXTURES_REQUIRED UnstructRestart)
endif()

if (ALBANY_IFPACK2)
  # Normal run
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_unstructT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_unstructT.yaml)
  add_test(${testName}_Tpetra ${Albany.exe} input_fo_gis_unstructT.yaml)
  set_tests_properties(${testName}_Tpetra
                       PROPERTIES
                       LABELS "LandIce;Tpetra;Forward"
                       FIXTURES_REQUIRED PopulateMeshes)

  # Memoization run
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_unstruct_memT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_unstruct_memT.yaml)
  add_test(${testName}_Memoization ${Albany.exe} input_fo_gis_unstruct_memT.yaml)
  set_tests_properties(${testName}_Memoization
                       PROPERTIES
                       LABELS "LandIce;Tpetra;Forward"
                       FIXTURES_REQUIRED PopulateMeshes)
endif()

if (NOT ALBANY_PARALELL_EXODUS AND ALBANY_MPI)
	set (testNameHumboldtDecompMesh ${testNameRoot}_Humboldt_decompMesh)
  add_test (NAME ${testNameHumboldtDecompMesh}
        COMMAND ${SEACAS_DECOMP} -processors ${MPIMNP} humboldt_2d.exo
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../AsciiMeshes/Humboldt)
  set_tests_properties (${testNameHumboldtDecompMesh} PROPERTIES FIXTURES_SETUP humboldtMeshSetup)
endif()

if (ALBANY_FROSCH)
  # Normal run
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_unstructT_FROSch.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_unstructT_FROSch.yaml)
  add_test(${testName}_Tpetra_FROSch ${Albany.exe} input_fo_gis_unstructT_FROSch.yaml)
  set_tests_properties(${testName}_Tpetra_FROSch
                       PROPERTIES
                       LABELS "LandIce;Tpetra;Forward"
                       FIXTURES_REQUIRED PopulateMeshes)
  
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_humboldt_frosch_fluxdiv.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_humboldt_frosch_fluxdiv.yaml)
  add_test(${testNameRoot}_Humboldt_FluxDiv_Tpetra_FROSch ${Albany.exe} input_fo_humboldt_frosch_fluxdiv.yaml)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_humboldt_frosch_power_law.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_humboldt_frosch_power_law.yaml)
  add_test(${testNameRoot}_Humboldt_powerLaw_Tpetra_FROSch ${Albany.exe} input_fo_humboldt_frosch_power_law.yaml)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_humboldt_frosch_effect_press.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_humboldt_frosch_effect_press.yaml)
  add_test(${testNameRoot}_Humboldt_effectPress_Tpetra_FROSch ${Albany.exe} input_fo_humboldt_frosch_effect_press.yaml)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_humboldt_frosch_pressurized_bed.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/input_fo_humboldt_frosch_pressurized_bed.yaml)
add_test(${testNameRoot}_Humboldt_pressurizedBed_Tpetra_FROSch ${Albany.exe} input_fo_humboldt_frosch_pressurized_bed.yaml)

  if (NOT ALBANY_PARALELL_EXODUS AND ALBANY_MPI)
    set_tests_properties(${testNameRoot}_Humboldt_FluxDiv_Tpetra_FROSch
                       PROPERTIES
                       LABELS            "LandIce;Tpetra;Forward"
                       FIXTURES_REQUIRED humboldtMeshSetup)
                       
    set_tests_properties(${testNameRoot}_Humboldt_powerLaw_Tpetra_FROSch
                       PROPERTIES
                       LABELS            "LandIce;Tpetra;Forward"
                       FIXTURES_REQUIRED humboldtMeshSetup)

    set_tests_properties(${testNameRoot}_Humboldt_effectPress_Tpetra_FROSch
                       PROPERTIES
                       LABELS            "LandIce;Tpetra;Forward"
                       FIXTURES_REQUIRED humboldtMeshSetup)
                       
  else ()
    set_tests_properties(${testNameRoot}_Humboldt_FluxDiv_Tpetra_FROSch
                       PROPERTIES
                       LABELS         "LandIce;Tpetra;Forward")
                       
    set_tests_properties(${testNameRoot}_Humboldt_powerLaw_Tpetra_FROSch
                       PROPERTIES
                       LABELS         "LandIce;Tpetra;Forward")
                       
    set_tests_properties(${testNameRoot}_Humboldt_effectPress_Tpetra_FROSch
                       PROPERTIES
                       LABELS         "LandIce;Tpetra;Forward")
  endif()
  
endif()

if (ALBANY_MUELU)
  # Normal run
  
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_humboldt_muelu.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_humboldt_muelu.yaml)
  add_test(${testNameRoot}_Humboldt_Tpetra_MueLu ${Albany.exe} input_fo_humboldt_muelu.yaml)

  if (NOT ALBANY_PARALELL_EXODUS AND ALBANY_MPI)
    set_tests_properties(${testNameRoot}_Humboldt_Tpetra_MueLu
                       PROPERTIES
                       LABELS            "LandIce;Epetra;Forward"
                       FIXTURES_REQUIRED humboldtMeshSetup)
  else ()
    set_tests_properties(${testNameRoot}_Humboldt_Tpetra_MueLu
                       PROPERTIES
                       LABELS         "LandIce;Epetra;Forward")
  endif()
  
endif()


####################################
###          20km tests          ###
####################################

set (testName ${testNameRoot}_Gis20km)
if (ALBANY_EPETRA) 
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis20km_test.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis20km_test.yaml)

  add_test(${testName}_Epetra ${Albany.exe} input_fo_gis20km_test.yaml)
  set_tests_properties(${testName}_Epetra  PROPERTIES LABELS "LandIce;Epetra;Forward")
endif()

if (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis20km_testT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis20km_testT.yaml)

  add_test(${testName}_Tpetra ${Albany.exe} input_fo_gis20km_testT.yaml)
  set_tests_properties(${testName}_Tpetra  PROPERTIES LABELS "LandIce;Tpetra;Forward")
endif()

####################################
###    Coupled Thickness tests   ###
####################################

set (testName ${testNameRoot}_CoupledThickness)

if (ALBANY_EPETRA)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_coupled.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_coupled.yaml)

  add_test(${testName}_Epetra ${Albany.exe} input_fo_gis_coupled.yaml)
  set_tests_properties(${testName}_Epetra  PROPERTIES LABELS "LandIce;Epetra;Forward")
endif()

IF(NOT ALBANY_ENABLE_CUDA)
  #IKT, 3/14/19: the following tests need to be debugged in CUDA build and turned back on; 
  #see issue #420. 
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_coupledT.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_coupledT.yaml)

  add_test(${testName}_Tpetra ${Albany.exe} input_fo_gis_coupledT.yaml)
  set_tests_properties(${testName}_Tpetra  PROPERTIES LABELS "LandIce;Tpetra;Forward")
endif()

####################################
###  Adjoint Sensitivity tests   ###
####################################

set (testName ${testNameRoot}_AdjointSensitivity)

# Tetra
if (ALBANY_EPETRA)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_adjoint_sensitivity.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_adjoint_sensitivity.yaml)
  add_test(${testName}_Epetra ${Albany.exe} input_fo_gis_adjoint_sensitivity.yaml)
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "LandIce;Epetra;Forward")
endif()

if (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_adjoint_sensitivityT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_adjoint_sensitivityT.yaml)
  add_test(${testName}_Tpetra ${Albany.exe} input_fo_gis_adjoint_sensitivityT.yaml)
  set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "LandIce;Tpetra;Forward")
  set (testName ${testNameRoot}_ForwardSensitivity)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_forward_sensitivityT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_forward_sensitivityT.yaml)
  add_test(${testName}_Tpetra ${Albany.exe} input_fo_gis_forward_sensitivityT.yaml)
  set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "LandIce;Tpetra;Forward")
endif()

# Wedge
set (testName ${testNameRoot}_AdjointSensitivity_Wedge)

if (ALBANY_EPETRA)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_wedge_adjoint_sensitivity.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_wedge_adjoint_sensitivity.yaml)
  add_test(${testName} ${Albany.exe} input_fo_gis_wedge_adjoint_sensitivity.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Epetra;Forward")
endif()

# Basal friction sensitivity
set (testName ${testNameRoot}_AdjointSensitivity_BasalFriction_FluxDivergence)

if (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_betaT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_betaT.yaml)
  add_test(${testName}_Tpetra ${Albany.exe} input_fo_gis_analysis_betaT.yaml)
  set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "LandIce;Tpetra;Forward")

  if (ALBANY_ROL)
    set (testName ${testNameRoot}_Analysis_BasalFriction)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_betaT.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_betaT.yaml)
    add_test(${testName} ${AlbanyAnalysis.exe} input_fo_gis_analysis_betaT.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Tpetra;Analysis;ROL")

    set (testName ${testNameRoot}_Analysis_BasalFriction_Memoization)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_beta_memT.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_beta_memT.yaml)
    add_test(${testName} ${AlbanyAnalysis.exe} input_fo_gis_analysis_beta_memT.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Tpetra;Analysis;ROL")
   
    if(ALBANY_TEKO)
      set (testName ${testNameRoot}_Analysis_BasalFriction_Hessian)
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_beta_hessianT.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_beta_hessianT.yaml)
      add_test(${testName} ${AlbanyAnalysis.exe} input_fo_gis_analysis_beta_hessianT.yaml)
      set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Tpetra;Analysis;ROL")
      
      set (testName ${testNameRoot}_Analysis_BasalFriction_Hessian_matrix_free)
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_beta_hessian_matfreeT.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_beta_hessian_matfreeT.yaml)
      add_test(${testName} ${AlbanyAnalysis.exe} input_fo_gis_analysis_beta_hessian_matfreeT.yaml)
      set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Tpetra;Analysis;ROL")
    endif()
  endif()
endif()

# Basal friction and stiffening factor sensitivity
set (testName ${testNameRoot}_AdjointSensitivity_StiffeningBasalFriction)

if (ALBANY_EPETRA)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_stiffening.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_stiffening.yaml)
  add_test(${testName}_Epetra ${Albany.exe} input_fo_gis_analysis_stiffening.yaml)
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "LandIce;Epetra;Forward")
endif()

if (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_stiffeningT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_stiffeningT.yaml)
  add_test(${testName}_Tpetra ${Albany.exe} input_fo_gis_analysis_stiffeningT.yaml)
  set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "LandIce;Tpetra;Forward")

  if (ALBANY_ROL AND ALBANY_TEKO)
    set (testName ${testNameRoot}_Analysis_StiffeningBasalFriction)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_stiffeningT.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_stiffeningT.yaml)
    add_test(${testName} ${AlbanyAnalysis.exe} input_fo_gis_analysis_stiffeningT.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Tpetra;Analysis;ROL")

    set (testName ${testNameRoot}_Analysis_StiffeningBasalFriction_Memoization)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_stiffening_memT.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_stiffening_memT.yaml)
    add_test(${testName} ${AlbanyAnalysis.exe} input_fo_gis_analysis_stiffening_memT.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Tpetra;Analysis;ROL")
  endif()
endif()

if (ALBANY_MESH_DEPENDS_ON_PARAMETERS)
  # Thickness sensitivity
  set (testName ${testNameRoot}_AdjointSensitivity_Thickness)
  if (ALBANY_EPETRA)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_adjoint_sensitivity_thickness.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_adjoint_sensitivity_thickness.yaml)
    add_test(${testName}_Epetra ${Albany.exe} input_fo_gis_adjoint_sensitivity_thickness.yaml)
    set_tests_properties(${testName}_Epetra PROPERTIES LABELS "LandIce;Epetra;Forward")
  endif()

  if (ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_adjoint_sensitivity_thicknessT.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_adjoint_sensitivity_thicknessT.yaml)
    add_test(${testName}_Tpetra ${Albany.exe} input_fo_gis_adjoint_sensitivity_thicknessT.yaml)
    set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "LandIce;Tpetra;Forward")
    
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_adjoint_sensitivity_bed_and_topT.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_adjoint_sensitivity_bed_and_topT.yaml)
    add_test(${testName}_MoveSurfHeightAndBed ${Albany.exe} input_fo_gis_adjoint_sensitivity_bed_and_topT.yaml)
    set_tests_properties(${testName}_MoveSurfHeightAndBed PROPERTIES LABELS "LandIce;Tpetra;Forward")

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_adjoint_sensitivity_thickness_adjustSurfHeightT.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_adjoint_sensitivity_thickness_adjustSurfHeightT.yaml)
    add_test(${testName}_MoveSurfHeight ${Albany.exe} input_fo_gis_adjoint_sensitivity_thickness_adjustSurfHeightT.yaml)
    set_tests_properties(${testName}_MoveSurfHeight PROPERTIES LABELS "LandIce;Tpetra;Forward")
  endif()

  # Two parameters sensitivity
  set (testName ${testNameRoot}_AdjointSensitivity_TwoParameters)
  if (ALBANY_EPETRA)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_two_params.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_two_params.yaml)
    add_test(${testName}_Epetra ${Albany.exe} input_fo_gis_analysis_two_params.yaml)
    set_tests_properties(${testName}_Epetra PROPERTIES LABELS "LandIce;Epetra;Forward")
  endif()

  if (ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_two_paramsT.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_two_paramsT.yaml)
    add_test(${testName}_Tpetra ${Albany.exe} input_fo_gis_analysis_two_paramsT.yaml)
    set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "LandIce;Tpetra;Forward")
  endif()
endif()

####################################
###     Sensitivity SMB tests    ###
####################################

set (testName ${testNameRoot}_SensSMBwrtBeta)

if (ALBANY_EPETRA)
  # First run
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_beta_smb.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_beta_smb.yaml)
  add_test(${testName}_Epetra ${Albany.exe} input_fo_gis_beta_smb.yaml)
  set_tests_properties(${testName}_Epetra PROPERTIES RUN_SERIAL TRUE LABELS "LandIce;Epetra;Forward")

  # Restart
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_beta_smb_restart.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_beta_smb_restart.yaml)

  add_test(${testName}_Restart ${Albany.exe} input_fo_gis_beta_smb_restart.yaml)
  set_tests_properties(${testName}_Restart  PROPERTIES RUN_SERIAL TRUE DEPENDS ${testName}_Epetra)
  set_tests_properties(${testName}_Epetra PROPERTIES RUN_SERIAL TRUE LABELS "LandIce;Epetra;Forward")
endif()

if (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_beta_smbT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_beta_smbT.yaml)
  add_test(${testName}_Tpetra ${Albany.exe} input_fo_gis_beta_smbT.yaml)
  set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "LandIce;Tpetra;Forward")
endif()

####################################
###        Manifold tests        ###
####################################

if (ALBANY_EPETRA)
  set (testName ${testNameRoot}_Manifold)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_manifold.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_manifold.yaml)

  add_test(${testName} ${Albany.exe} input_fo_gis_manifold.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Epetra;Forward")
endif()

####################################
###   Laplacian sampling tests   ###
####################################

if (ALBANY_EPETRA)
  set (testName ${testNameRoot}_LaplacianPriorSampling)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_gis_sampling.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_gis_sampling.yaml)
  add_test(${testName} ${Albany.exe} input_gis_sampling.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Epetra;Forward")
endif()


if (ALBANY_LARGE_TESTS)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis20km_refine.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis20km_refine.yaml COPYONLY)

  if(ALBANY_STK_PERCEPT AND ALBANY_EPETRA)
    add_test(${testName}_Gis20km_refine ${Albany8.exe} input_fo_gis20km_refine.yaml)
  endif()
endif()
