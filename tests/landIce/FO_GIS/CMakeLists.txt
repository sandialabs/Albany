# Name the test with the directory name

get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${parentDirName}_${dirName})

# This variable will be set in the input files during the 'configure_file' call
if (ALBANY_PARALELL_EXODUS)
  set (USE_SERIAL_MESH "true")
else ()
  # If there's no Iopx, we *must* have ALBANY_DECOMP defined (to decompose the mesh)
  # OR execute with only one rank.
  IF (NOT ${ALBANY_SEACAS_PATH})
    MESSAGE (FATAL_ERROR "Error! Cannot execute FO_GIS tests in parallel without Iopx or decomp from Trilinos.\n")
  ENDIF()
 set (USE_SERIAL_MESH "false")
endif()

# This variable will be set in the input files during the 'configure_file' call
if (ALBANY_AMESOS2_SUPERLUDIST)
  set (DIRECT_SOLVER "superlu_dist")
elseif (ALBANY_AMESOS2_SUPERLU)
  set (DIRECT_SOLVER "superlu")
else ()
  set (DIRECT_SOLVER "klu")
endif()


###########################################
###     Import 2d ascii mesh (test)     ###
###########################################

set (testName ${testNameRoot}_Import2DAsciiMesh)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_create_exo_from_msh.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_create_exo_from_msh.yaml)
 
IF (ALBANY_PARALELL_EXODUS)
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
IF (NOT ALBANY_PARALELL_EXODUS)
  set (testNameDecompMesh ${testNameRoot}_decompMesh)
  add_test (NAME ${testNameDecompMesh}
            COMMAND ${SerialSeacasDecomp.exe} -processors ${MPIMNP} gis_unstruct_2d.exo
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../ExoMeshes)
  set_tests_properties (${testNameDecompMesh} PROPERTIES FIXTURES_SETUP meshSetup RUN_SERIAL TRUE)

  set_tests_properties(${testName}
                       PROPERTIES
                       LABELS            "LandIce;Forward"
                       FIXTURES_SETUP    PopulateMeshes
                       FIXTURES_REQUIRED meshSetup)
ELSE ()
  set_tests_properties(${testName}
                       PROPERTIES
                       LABELS         "LandIce;Forward"
                       FIXTURES_SETUP PopulateMeshes)
ENDIF()

####################################
###    Populate mesh2d (test)    ###
####################################

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_populate_mesh_2d.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_populate_mesh_2d.yaml)
set (testName ${testNameRoot}_PopulateMesh2d)
add_test(${testName} ${Albany.exe} input_fo_gis_populate_mesh_2d.yaml)

# If Iopx is not present, we must decompose the input mesh first
IF (NOT ALBANY_PARALELL_EXODUS)
  set_tests_properties(${testName}
                       PROPERTIES
                       LABELS            "LandIce;Forward"
                       FIXTURES_SETUP    PopulateMesh2d
                       FIXTURES_REQUIRED meshSetup)
ELSE ()
  set_tests_properties(${testName}
                       PROPERTIES
                       LABELS         "LandIce;Forward"
                       FIXTURES_SETUP PopulateMesh2d)
ENDIF()

####################################
###        Unstruct tests        ###
####################################

set (testName ${testNameRoot}_Unstructured)

if (ALBANY_IFPACK2)
  # Normal run
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_unstruct.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_unstruct.yaml)
  add_test(${testName} ${Albany.exe} input_fo_gis_unstruct.yaml)
  set_tests_properties(${testName}
                       PROPERTIES
                       LABELS "LandIce;Forward"
                       FIXTURES_REQUIRED PopulateMeshes
                       FIXTURES_SETUP ${testName})

  # Logically extruded mesh
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_unstruct_extruded.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_unstruct_extruded.yaml)
  add_test(${testName}_extruded ${Albany.exe} input_fo_gis_unstruct_extruded.yaml)
  set_tests_properties(${testName}_extruded
                       PROPERTIES
                       LABELS "LandIce;Forward"
                       FIXTURES_REQUIRED PopulateMesh2d
                       FIXTURES_SETUP ${testName}_extruded)

  # Memoization run
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_unstruct_mem.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_unstruct_mem.yaml)
  add_test(${testName}_Memoization ${Albany.exe} input_fo_gis_unstruct_mem.yaml)
  set_tests_properties(${testName}_Memoization
                       PROPERTIES
                       LABELS "LandIce;Forward"
                       FIXTURES_REQUIRED PopulateMeshes)

  if (NOT ALBANY_DISABLE_OUTPUT_MESH)
    # Restart
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_unstruct_restart.yaml
                    ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_unstruct_restart.yaml)
    add_test(${testName}_Restart ${Albany.exe} input_fo_gis_unstruct_restart.yaml)
    set_tests_properties(${testName}_Restart
                          PROPERTIES
                          LABELS "LandIce;Forward"
                          RUN_SERIAL TRUE
                          FIXTURES_REQUIRED ${testName})

    # Restart the logically extruded mesh test
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_unstruct_extruded_restart.yaml
                    ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_unstruct_extruded_restart.yaml)
    add_test(${testName}_extruded_restart ${Albany.exe} input_fo_gis_unstruct_extruded_restart.yaml)
    set_tests_properties(${testName}_extruded_restart
                          PROPERTIES
                          LABELS "LandIce;Forward"
                          RUN_SERIAL TRUE
                          FIXTURES_REQUIRED ${testName}_extruded)
  endif()
endif()

if (NOT ALBANY_PARALELL_EXODUS)
	set (testNameHumboldtDecompMesh ${testNameRoot}_Humboldt_2d_decompMesh)
  add_test (NAME ${testNameHumboldtDecompMesh}
        COMMAND ${SerialSeacasDecomp.exe} -processors ${MPIMNP} humboldt_2d.exo
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../AsciiMeshes/Humboldt)
  set_tests_properties (${testNameHumboldtDecompMesh} PROPERTIES FIXTURES_SETUP humboldtMeshSetup2d)

	set (testNameHumboldtDecompMesh ${testNameRoot}_Humboldt_contiguous_2d_decompMesh)
  add_test (NAME ${testNameHumboldtDecompMesh}
        COMMAND ${SerialSeacasDecomp.exe} -processors ${MPIMNP} humboldt_contiguous_2d.exo
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../AsciiMeshes/Humboldt)
  set_tests_properties (${testNameHumboldtDecompMesh} PROPERTIES FIXTURES_SETUP humboldtMeshSetupContiguous2d)
endif()

if (ALBANY_FROSCH)
  # Normal run
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_unstruct_FROSch.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_unstruct_FROSch.yaml)
  add_test(${testName}_FROSch ${Albany.exe} input_fo_gis_unstruct_FROSch.yaml)
  set_tests_properties(${testName}_FROSch
                       PROPERTIES
                       LABELS "LandIce;Forward"
                       FIXTURES_REQUIRED PopulateMeshes)
  
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_humboldt_frosch_fluxdiv.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_humboldt_frosch_fluxdiv.yaml)
  add_test(${testNameRoot}_Humboldt_FluxDiv_FROSch ${Albany.exe} input_fo_humboldt_frosch_fluxdiv.yaml)
  set_tests_properties(${testNameRoot}_Humboldt_FluxDiv_FROSch
                     PROPERTIES
                     LABELS         "LandIce;Forward")

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_humboldt_frosch_power_law.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_humboldt_frosch_power_law.yaml)
  add_test(${testNameRoot}_Humboldt_powerLaw_FROSch ${Albany.exe} input_fo_humboldt_frosch_power_law.yaml)
  set_tests_properties(${testNameRoot}_Humboldt_powerLaw_FROSch
                     PROPERTIES
                     LABELS         "LandIce;Forward")

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_humboldt_frosch_effect_press.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_humboldt_frosch_effect_press.yaml)
  add_test(${testNameRoot}_Humboldt_effectPress_FROSch ${Albany.exe} input_fo_humboldt_frosch_effect_press.yaml)
  set_tests_properties(${testNameRoot}_Humboldt_effectPress_FROSch
                     PROPERTIES
                     LABELS         "LandIce;Forward")

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_humboldt_frosch_pressurized_bed.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_humboldt_frosch_pressurized_bed.yaml)
  add_test(${testNameRoot}_Humboldt_pressurizedBed_FROSch ${Albany.exe} input_fo_humboldt_frosch_pressurized_bed.yaml)
  set_tests_properties(${testNameRoot}_Humboldt_pressurizedBed_FROSch
                     PROPERTIES
                     LABELS            "LandIce;Forward")
                       
  if (NOT ALBANY_PARALELL_EXODUS)
    set_tests_properties(${testNameRoot}_Humboldt_FluxDiv_FROSch
      PROPERTIES FIXTURES_REQUIRED humboldtMeshSetupContiguous2d)

    set_tests_properties(${testNameRoot}_Humboldt_pressurizedBed_FROSch
      PROPERTIES FIXTURES_REQUIRED humboldtMeshSetup2d)
                       
    set_tests_properties(${testNameRoot}_Humboldt_powerLaw_FROSch
      PROPERTIES FIXTURES_REQUIRED humboldtMeshSetup2d)

    set_tests_properties(${testNameRoot}_Humboldt_effectPress_FROSch
      PROPERTIES FIXTURES_REQUIRED humboldtMeshSetup2d)
                       
  endif()
  
endif()

if (ALBANY_MUELU)
  # Normal run
  
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_humboldt_muelu.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_humboldt_muelu.yaml)
  add_test(${testNameRoot}_Humboldt_MueLu ${Albany.exe} input_fo_humboldt_muelu.yaml)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_humboldt_depthInt_muelu.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_humboldt_depthInt_muelu.yaml)
  add_test(${testNameRoot}_Humboldt_DepthIntegrated_MueLu ${Albany.exe} input_fo_humboldt_depthInt_muelu.yaml)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_humboldt_depthInt_regCoulomb.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/input_fo_humboldt_depthInt_regCoulomb.yaml)
  add_test(${testNameRoot}_Humboldt_DepthIntegrated_RegCoulomb ${Albany.exe} input_fo_humboldt_depthInt_regCoulomb.yaml)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_humboldt_depthInt_regCoulomb_constA.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/input_fo_humboldt_depthInt_regCoulomb_constA.yaml)
  add_test(${testNameRoot}_Humboldt_DepthIntegrated_RegCoulomb_ConstA ${Albany.exe} input_fo_humboldt_depthInt_regCoulomb_constA.yaml)

  if (NOT ALBANY_PARALELL_EXODUS)
    set_tests_properties(${testNameRoot}_Humboldt_MueLu
                       PROPERTIES
                       LABELS            "LandIce;Forward"
                       FIXTURES_REQUIRED humboldtMeshSetup)

    set_tests_properties(${testNameRoot}_Humboldt_DepthIntegrated_MueLu
                       PROPERTIES
                       LABELS            "LandIce;Forward"
                       FIXTURES_REQUIRED humboldtMeshSetup)
    set_tests_properties(${testNameRoot}_Humboldt_DepthIntegrated_RegCoulomb
                       PROPERTIES
                       LABELS            "LandIce;Forward"
                       FIXTURES_REQUIRED humboldtMeshSetup)
    set_tests_properties(${testNameRoot}_Humboldt_DepthIntegrated_RegCoulomb_ConstA
                       PROPERTIES
                       LABELS            "LandIce;Forward"
                       FIXTURES_REQUIRED humboldtMeshSetup)
  else ()
    set_tests_properties(${testNameRoot}_Humboldt_MueLu
                       PROPERTIES
                       LABELS         "LandIce;Forward")

    set_tests_properties(${testNameRoot}_Humboldt_DepthIntegrated_MueLu
                       PROPERTIES
                       LABELS         "LandIce;Forward")

    set_tests_properties(${testNameRoot}_Humboldt_DepthIntegrated_RegCoulomb
                       PROPERTIES
                       LABELS         "LandIce;Forward")

    set_tests_properties(${testNameRoot}_Humboldt_DepthIntegrated_RegCoulomb_ConstA
                       PROPERTIES
                       LABELS         "LandIce;Forward")
  endif()
  
endif()


####################################
###          20km tests          ###
####################################

set (testName ${testNameRoot}_Gis20km)

if (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis20km_test.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis20km_test.yaml)

  add_test(${testName} ${Albany.exe} input_fo_gis20km_test.yaml)
  set_tests_properties(${testName}  PROPERTIES LABELS "LandIce;Forward")
endif()

####################################
###    Coupled Thickness tests   ###
####################################

IF(NOT Kokkos_ENABLE_CUDA)
  #IKT, 3/14/19: the following tests need to be debugged in CUDA build and turned back on; 
  #see issue #420. 
  set (testName ${testNameRoot}_CoupledThickness)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_coupled.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_coupled.yaml)

  add_test(${testName} ${Albany.exe} input_fo_gis_coupled.yaml)
  set_tests_properties(${testName}  PROPERTIES LABELS "LandIce;Forward")

  if (ALBANY_MESH_DEPENDS_ON_SOLUTION)
    set (testName ${testNameRoot}_Humboldt_Transient_Sensitivities)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_humboldt_transient_sensitivities.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/input_fo_humboldt_transient_sensitivities.yaml)

    add_test(${testName} ${Albany.exe} input_fo_humboldt_transient_sensitivities.yaml)
    set_tests_properties(${testName}  PROPERTIES LABELS "LandIce;Forward" FIXTURES_REQUIRED humboldtMeshSetup)
  
  else()
    set (testName ${testNameRoot}_Humboldt_Transient)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_humboldt_transient.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_fo_humboldt_transient.yaml)
    add_test(${testName} ${Albany.exe} input_fo_humboldt_transient.yaml)
    set_tests_properties(${testName}  PROPERTIES LABELS "LandIce;Forward" FIXTURES_REQUIRED humboldtMeshSetup)
  endif()

endif()

####################################
###  Adjoint Sensitivity tests   ###
####################################

set (testName ${testNameRoot}_AdjointSensitivity)

if (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_adjoint_sensitivity.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_adjoint_sensitivity.yaml)
  add_test(${testName} ${Albany.exe} input_fo_gis_adjoint_sensitivity.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward")
  set (testName ${testNameRoot}_ForwardSensitivity)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_forward_sensitivity.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_forward_sensitivity.yaml)
  add_test(${testName} ${Albany.exe} input_fo_gis_forward_sensitivity.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward")
endif()

# Basal friction sensitivity
set (testName ${testNameRoot}_AdjointSensitivity_BasalFriction_FluxDivergence)

if (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_beta.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_beta.yaml)
  add_test(${testName} ${Albany.exe} input_fo_gis_analysis_beta.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward")

  if (ALBANY_ROL)
    set (testName ${testNameRoot}_Analysis_BasalFriction)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_beta.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_beta.yaml)
    add_test(${testName} ${AlbanyAnalysis.exe} input_fo_gis_analysis_beta.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Analysis;ROL")

    set (testName ${testNameRoot}_Analysis_BasalFriction_Memoization)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_beta_mem.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_beta_mem.yaml)
    add_test(${testName} ${AlbanyAnalysis.exe} input_fo_gis_analysis_beta_mem.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Analysis;ROL")
   
    if(ALBANY_TEKO)
      set (testName ${testNameRoot}_Analysis_BasalFriction_Hessian)
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_beta_hessian.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_beta_hessian.yaml)
      add_test(${testName} ${AlbanyAnalysis.exe} input_fo_gis_analysis_beta_hessian.yaml)
      set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Analysis;ROL")
      
      set (testName ${testNameRoot}_Analysis_BasalFriction_Hessian_matrix_free)
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_beta_hessian_matfree.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_beta_hessian_matfree.yaml)
      add_test(${testName} ${AlbanyAnalysis.exe} input_fo_gis_analysis_beta_hessian_matfree.yaml)
      set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Analysis;ROL")
    endif()

    if(ALBANY_TEKO AND ALBANY_MUELU)
      set (testName ${testNameRoot}_Humboldt_Analysis)
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_humboldt_analysis.yaml
                     ${CMAKE_CURRENT_BINARY_DIR}/input_fo_humboldt_analysis.yaml)
      add_test(${testName} ${AlbanyAnalysis.exe} input_fo_humboldt_analysis.yaml)

      set (testName ${testNameRoot}_Humboldt_Analysis_Mat_Free_Reg)
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_humboldt_analysis_mat_free_reg.yaml
                     ${CMAKE_CURRENT_BINARY_DIR}/input_fo_humboldt_analysis_mat_free_reg.yaml)
      add_test(${testName} ${AlbanyAnalysis.exe} input_fo_humboldt_analysis_mat_free_reg.yaml)
                         
      if (NOT ALBANY_PARALELL_EXODUS)
        set_tests_properties(${testNameRoot}_Humboldt_Analysis
          PROPERTIES FIXTURES_REQUIRED humboldtMeshSetupContiguous2d)
        set_tests_properties(${testNameRoot}_Humboldt_Analysis_Mat_Free_Reg
          PROPERTIES FIXTURES_REQUIRED humboldtMeshSetupContiguous2d)
      endif()
    endif()

  endif()
endif()

# Basal friction and stiffening factor sensitivity
set (testName ${testNameRoot}_AdjointSensitivity_StiffeningBasalFriction)

if (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_stiffening.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_stiffening.yaml)
  add_test(${testName} ${Albany.exe} input_fo_gis_analysis_stiffening.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward")

  if (ALBANY_ROL AND ALBANY_TEKO)
    set (testName ${testNameRoot}_Analysis_StiffeningBasalFriction)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_stiffening.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_stiffening.yaml)
    add_test(${testName} ${AlbanyAnalysis.exe} input_fo_gis_analysis_stiffening.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Analysis;ROL")

    set (testName ${testNameRoot}_Analysis_StiffeningBasalFriction_Memoization)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_stiffening_mem.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_stiffening_mem.yaml)
    add_test(${testName} ${AlbanyAnalysis.exe} input_fo_gis_analysis_stiffening_mem.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Analysis;ROL")
  endif()
endif()

if (ALBANY_MESH_DEPENDS_ON_PARAMETERS)
  # Thickness sensitivity
  set (testName ${testNameRoot}_AdjointSensitivity_Thickness)

  if (ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_adjoint_sensitivity_thickness.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_adjoint_sensitivity_thickness.yaml)
    add_test(${testName} ${Albany.exe} input_fo_gis_adjoint_sensitivity_thickness.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward")
    
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_adjoint_sensitivity_bed_and_top.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_adjoint_sensitivity_bed_and_top.yaml)
    add_test(${testName}_MoveSurfHeightAndBed ${Albany.exe} input_fo_gis_adjoint_sensitivity_bed_and_top.yaml)
    set_tests_properties(${testName}_MoveSurfHeightAndBed PROPERTIES LABELS "LandIce;Forward")

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_adjoint_sensitivity_thickness_adjustSurfHeight.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_adjoint_sensitivity_thickness_adjustSurfHeight.yaml)
    add_test(${testName}_MoveSurfHeight ${Albany.exe} input_fo_gis_adjoint_sensitivity_thickness_adjustSurfHeight.yaml)
    set_tests_properties(${testName}_MoveSurfHeight PROPERTIES LABELS "LandIce;Forward")
  endif()

  # Two parameters sensitivity
  set (testName ${testNameRoot}_AdjointSensitivity_TwoParameters)

  if (ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_two_params.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_two_params.yaml)
    add_test(${testName} ${Albany.exe} input_fo_gis_analysis_two_params.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward")
  endif()

  set (testName ${testNameRoot}_Humboldt_Analysis_ThreeParameters)
  if (ALBANY_IFPACK2 AND ALBANY_TEKO AND ALBANY_LARGE_TESTS)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_humboldt_analysis_three_params.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_humboldt_analysis_three_params.yaml)
    add_test(${testName} ${AlbanyAnalysis.exe} input_fo_humboldt_analysis_three_params.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Large;Analysis")
    if (NOT ALBANY_PARALELL_EXODUS)
      set_tests_properties(${testNameRoot}_Humboldt_Analysis_ThreeParameters
        PROPERTIES FIXTURES_REQUIRED humboldtMeshSetupContiguous2d)
    endif()
  endif()
endif()

####################################
###     Sensitivity SMB tests    ###
####################################

set (testName ${testNameRoot}_SensSMBwrtBeta)

if (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_beta_smb.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_beta_smb.yaml)
  add_test(${testName} ${Albany.exe} input_fo_gis_beta_smb.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward")

  if (NOT ALBANY_DISABLE_OUTPUT_MESH)
    # Restart
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_beta_smb_restart.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_beta_smb_restart.yaml)

    add_test(${testName}_Restart ${Albany.exe} input_fo_gis_beta_smb_restart.yaml)
    set_tests_properties(${testName}_Restart  PROPERTIES RUN_SERIAL TRUE DEPENDS ${testName})
    set_tests_properties(${testName} PROPERTIES RUN_SERIAL TRUE LABELS "LandIce;Forward")
  endif()
endif()

####################################
###        Manifold tests        ###
####################################

set (testName ${testNameRoot}_Manifold)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_manifold.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_manifold.yaml)

add_test(${testName} ${Albany.exe} input_fo_gis_manifold.yaml)
set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward")

####################################
###   Laplacian sampling tests   ###
####################################

if (ALBANY_IFPACK2)
  set (testName ${testNameRoot}_LaplacianPriorSampling)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_gis_sampling.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_gis_sampling.yaml)
  add_test(${testName} ${Albany.exe} input_gis_sampling.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward")

  set (testName ${testNameRoot}_Humboldt_LaplacianPriorSampling)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_humboldt_sampling.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_humboldt_sampling.yaml)
  add_test(${testName} ${Albany.exe} input_humboldt_sampling.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward")

  set (testName ${testNameRoot}_Humboldt_LaplacianPriorSampling_3D)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_humboldt_sampling_3D.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_humboldt_sampling_3D.yaml)
  add_test(${testName} ${Albany.exe} input_humboldt_sampling_3D.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward")


  if (NOT ALBANY_PARALELL_EXODUS)
    set_tests_properties(${testNameRoot}_Humboldt_LaplacianPriorSampling
      PROPERTIES FIXTURES_REQUIRED humboldtMeshSetupContiguous2d)

    set_tests_properties(${testNameRoot}_Humboldt_LaplacianPriorSampling_3D
      PROPERTIES FIXTURES_REQUIRED humboldtMeshSetupContiguous2d)
  endif()

endif()
