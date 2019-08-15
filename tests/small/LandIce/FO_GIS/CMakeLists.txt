# Name the test with the directory name
get_filename_component(testNameRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# This variable will be set in the input files during the 'configure_file' call
if (ALBANY_IOPX)
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

####################################
###     Populate mesh (test)     ###
####################################

set (testName ${testNameRoot}_PopulateMeshes)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_populate_meshes.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_populate_meshes.yaml)

add_test(${testName} ${Albany.exe} input_fo_gis_populate_meshes.yaml)
# If Iopx is not present, we must decompose the input mesh first
IF (NOT ALBANY_IOPX AND ALBANY_MPI)
  add_test (NAME FO_GIS_decompMesh
            COMMAND ${SEACAS_DECOMP} -processors ${MPIMNP} gis_unstruct_2d.exo
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../ExoMeshes)
  set_tests_properties (FO_GIS_decompMesh PROPERTIES FIXTURES_SETUP meshSetup)

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
                       FIXTURES_REQUIRED PopulateMeshes)

  # Restart
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_unstruct_restart.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_unstruct_restart.yaml)
  add_test(${testName}_Restart ${Albany.exe} input_fo_gis_unstruct_restart.yaml)
  set_tests_properties(${testName}_Restart
                       PROPERTIES
                       LABELS "LandIce;Epetra;Forward"
                       FIXTURES_REQUIRED PopulateMeshes)
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

  # Performance run
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_unstruct_perfT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_unstruct_perfT.yaml)
  add_test(${testName}_Performance ${Albany.exe} input_fo_gis_unstruct_perfT.yaml)
  set_tests_properties(${testName}_Performance
                       PROPERTIES
                       LABELS "LandIce;Tpetra;Forward"
                       FIXTURES_REQUIRED PopulateMeshes)
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

if (ALBANY_MESH_DEPENDS_ON_SOLUTION)
  # LB: Disabling this test for now, since ever since we merged #412 it no longer works
  #     To be clear, before merging #412 it passed, but it was computing the wrong thing...

  # set (testName ${testNameRoot}_CoupledThicknessShapeOpt)

  # if (ALBANY_EPETRA)
  #   configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_coupled_shape_opt.yaml
  #                  ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_coupled_shape_opt.yaml)

  #   add_test(${testName}_Epetra ${Albany.exe} input_fo_gis_coupled_shape_opt.yaml)
  #   set_tests_properties(${testName}_Epetra  PROPERTIES LABELS "LandIce;Epetra;Forward")
  # endif()
  # if (ALBANY_IFPACK2)
  #   configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_coupled_shape_optT.yaml
  #                  ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_coupled_shape_optT.yaml)

  #   add_test(${testName}_Tpetra ${Albany.exe} input_fo_gis_coupled_shape_optT.yaml)
  #   set_tests_properties(${testName}_Tpetra  PROPERTIES LABELS "LandIce;Tpetra;Forward")
  # endif()
else()
  set (testName ${testNameRoot}_CoupledThickness)

  if (ALBANY_EPETRA)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_coupled.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_coupled.yaml)

    add_test(${testName}_Epetra ${Albany.exe} input_fo_gis_coupled.yaml)
    set_tests_properties(${testName}_Epetra  PROPERTIES LABELS "LandIce;Epetra;Forward")
  endif()

  IF(NOT Kokkos_ENABLE_Cuda)
    #IKT, 3/14/19: the following tests need to be debugged in CUDA build and turned back on; 
    #see issue #420. 
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_coupledT.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_coupledT.yaml)

    add_test(${testName}_Tpetra ${Albany.exe} input_fo_gis_coupledT.yaml)
    set_tests_properties(${testName}_Tpetra  PROPERTIES LABELS "LandIce;Tpetra;Forward")
  endif()
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
set (testName ${testNameRoot}_AdjointSensitivity_BasalFriction)

if (ALBANY_EPETRA)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_beta.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_beta.yaml)
  add_test(${testName}_Epetra ${Albany.exe} input_fo_gis_analysis_beta.yaml)
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "LandIce;Epetra;Forward")
endif ()

if (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_betaT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_betaT.yaml)
  add_test(${testName}_Tpetra ${Albany.exe} input_fo_gis_analysis_betaT.yaml)
  set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "LandIce;Tpetra;Forward")
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
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "LandIce;Epetra;Forward")

  # Restart
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_beta_smb_restart.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_beta_smb_restart.yaml)

  add_test(${testName}_Restart ${Albany.exe} input_fo_gis_beta_smb_restart.yaml)
  set_tests_properties(${testName}_Restart  PROPERTIES DEPENDS ${testName}_Epetra)
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "LandIce;Epetra;Forward")
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

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_sampling.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_sampling.yaml)
  add_test(${testName} ${Albany.exe} input_fo_gis_sampling.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Epetra;Forward")
endif()
