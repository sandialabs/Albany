# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis20km_ascii.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis20km_ascii.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis20km.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis20km.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis10km_ascii.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis10km_ascii.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis10km.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis10km.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis5km_ascii.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis5km_ascii.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis5km.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis5km.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis2km_ascii.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis2km_ascii.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis2km.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis2km.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_populate_meshes.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_populate_meshes.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_unstruct.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_unstruct.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_unstructT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_unstructT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_unstruct_perfT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_unstruct_perfT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_coupled.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_coupled.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_coupledT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_coupledT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_coupled_shape_opt.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_coupled_shape_opt.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_coupled_shape_optT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_coupled_shape_optT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_manifold.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_manifold.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_unstruct_restart.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_unstruct_restart.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_adjoint_sensitivity.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_adjoint_sensitivity.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_adjoint_sensitivityT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_adjoint_sensitivityT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_wedge_adjoint_sensitivity.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_wedge_adjoint_sensitivity.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_adjoint_sensitivity_beta.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_adjoint_sensitivity_beta.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_adjoint_sensitivity_thickness.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_adjoint_sensitivity_thickness.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_adjoint_sensitivity_thicknessT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_adjoint_sensitivity_thicknessT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_adjoint_sensitivity_thickness_adjustSurfHeightT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_adjoint_sensitivity_thickness_adjustSurfHeightT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_stiffening.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_stiffening.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_stiffeningT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_stiffeningT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_beta.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_beta.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_betaT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_betaT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_thickness.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_thickness.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_two_params.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_two_params.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_two_paramsT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_two_paramsT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_beta_smb.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_beta_smb.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_beta_smbT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_beta_smbT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_beta_smb_restart.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_beta_smb_restart.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_wedge_pumi.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_wedge_pumi.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_sampling.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_sampling.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#3. Tests
#  First Test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis20km_test.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis20km_test.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis20km_testT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis20km_testT.yaml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_Gis20km ${Albany.exe} input_fo_gis20km_test.yaml)
set_tests_properties(${testName}_Gis20km  PROPERTIES LABELS CUDA_TEST)
ENDIF()
if (ALBANY_IOPX)
add_test(${testName}_GisPopulateMeshes ${AlbanyT.exe} input_fo_gis_populate_meshes.yaml)
ELSE()
add_test(NAME ${testName}_GisPopulateMeshes_DecompMesh
             COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
             -DTEST_NAME=GisPopulateMeshes_DecompMesh -DTEST_ARGS=input_fo_gis_populate_meshes.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
ENDIF()
if (ALBANY_EPETRA)
if (ALBANY_IOPX)
add_test(${testName}_GisUnstructured ${Albany.exe} input_fo_gis_unstruct.yaml)
set_tests_properties(${testName}_GisUnstructured PROPERTIES DEPENDS ${testName}_GisPopulateMeshes)
if (ALBANY_MESH_DEPENDS_ON_SOLUTION)
  add_test(${testName}_GisCoupledThicknessShapeOpt ${Albany.exe} input_fo_gis_coupled_shape_opt.yaml)
else()
  add_test(${testName}_GisCoupledThickness ${Albany.exe} input_fo_gis_coupled.yaml)
endif()
add_test(${testName}_GisManifold ${Albany.exe} input_fo_gis_manifold.yaml)
add_test(${testName}_GisRestartUnstructured ${Albany.exe} input_fo_gis_unstruct_restart.yaml)
add_test(${testName}_GisAdjointSensitivity ${Albany.exe} input_fo_gis_adjoint_sensitivity.yaml)
add_test(${testName}_GisWedgeAdjointSensitivity ${Albany.exe} input_fo_gis_wedge_adjoint_sensitivity.yaml)
add_test(${testName}_GisAdjointSensitivityBasalFriction ${Albany.exe} input_fo_gis_analysis_beta.yaml)
add_test(${testName}_GisAdjointSensitivityStiffeningBasalFriction ${Albany.exe} input_fo_gis_analysis_stiffening.yaml)
add_test(${testName}_GisLaplacianPriorSampling ${Albany.exe} input_fo_gis_sampling.yaml)
add_test(${testName}_GisSensSMBwrtBeta ${Albany.exe} input_fo_gis_beta_smb.yaml)
add_test(${testName}_GisSensSMBwrtBetaRestart ${Albany.exe} input_fo_gis_beta_smb_restart.yaml)
set_tests_properties(${testName}_GisSensSMBwrtBetaRestart  PROPERTIES DEPENDS ${testName}_GisSensSMBwrtBeta)
  if (ALBANY_MESH_DEPENDS_ON_PARAMETERS)
    add_test(${testName}_GisAdjointSensitivityThickness ${Albany.exe} input_fo_gis_adjoint_sensitivity_thickness.yaml)
    add_test(${testName}_GisAdjointSensitivityTwoParameters ${Albany.exe} input_fo_gis_analysis_two_params.yaml)
  endif()
set_tests_properties(${testName}_GisRestartUnstructured  PROPERTIES DEPENDS ${testName}_GisUnstructured)
else()
add_test(NAME ${testName}_GisUnstructured_DecompMesh
             COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
             -DTEST_NAME=GisUnstructured_DecompMesh -DTEST_ARGS=input_fo_gis_unstruct.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
set_tests_properties(${testName}_GisUnstructured_DecompMesh PROPERTIES DEPENDS ${testName}_GisPopulateMeshes_DecompMesh)
if (ALBANY_MESH_DEPENDS_ON_SOLUTION)
add_test(NAME ${testName}_GisCoupledThicknessShapeOpt_DecompMesh
             COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
             -DTEST_NAME=GisCoupledThicknessShapeOpt_DecompMesh 
             -DTEST_ARGS=input_fo_gis_coupled_shape_opt.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
else()
add_test(NAME ${testName}_GisCoupledThickness_DecompMesh
             COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
             -DTEST_NAME=GisCoupledThickness_DecompMesh 
             -DTEST_ARGS=input_fo_gis_coupled.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
endif()
add_test(NAME ${testName}_GisManifold_DecompMesh
             COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
             -DTEST_NAME=GisManifold_DecompMesh -DTEST_ARGS=input_fo_gis_manifold.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
add_test(NAME ${testName}_GisRestartUnstructured_DecompMesh
             COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
             -DTEST_NAME=GisRestartUnstructured_DecompMesh 
             -DTEST_ARGS=input_fo_gis_unstruct_restart.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
add_test(NAME ${testName}_GisAdjointSensitivity_DecompMesh
             COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
             -DTEST_NAME=GisAdjointSensitivity_DecompMesh 
             -DTEST_ARGS=input_fo_gis_adjoint_sensitivity.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
add_test(NAME ${testName}_GisWedgeAdjointSensitivity_DecompMesh
             COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
             -DTEST_NAME=GisWedgeAdjointSensitivity_DecompMesh 
             -DTEST_ARGS=input_fo_gis_wedge_adjoint_sensitivity.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
add_test(NAME ${testName}_GisAdjointSensitivityBasalFriction_DecompMesh
             COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
             -DTEST_NAME=GisAdjointSensitivityBasalFriction_DecompMesh 
             -DTEST_ARGS=input_fo_gis_analysis_beta.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
add_test(NAME ${testName}_GisAdjointSensitivityStiffeningBasalFriction_DecompMesh
             COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
             -DTEST_NAME=GisAdjointSensitivityStiffeningBasalFriction_DecompMesh 
             -DTEST_ARGS=input_fo_gis_analysis_stiffening.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
add_test(NAME ${testName}_GisLaplacianPriorSampling_DecompMesh
             COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
             -DTEST_NAME=GisLaplacianPriorSampling_DecompMesh 
             -DTEST_ARGS=input_fo_gis_sampling.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
add_test(NAME ${testName}_GisSensSMBwrtBeta_DecompMesh
             COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
             -DTEST_NAME=GisSensSMBwrtBeta_DecompMesh 
             -DTEST_ARGS=input_fo_gis_beta_smb.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
add_test(NAME ${testName}_GisSensSMBwrtBetaRestart_DecompMesh
             COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
             -DTEST_NAME=GisSensSMBwrtBetaRestart_DecompMesh 
             -DTEST_ARGS=input_fo_gis_beta_smb_restart.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
set_tests_properties(${testName}_GisSensSMBwrtBetaRestart_DecompMesh  PROPERTIES DEPENDS ${testName}_GisSensSMBwrtBeta_DecompMesh)
  if (ALBANY_MESH_DEPENDS_ON_PARAMETERS)
    add_test(NAME ${testName}_GisAdjointSensitivityThickness_DecompMesh
             COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
             -DTEST_NAME=GisAdjointSensitivityThickness_DecompMesh 
             -DTEST_ARGS=input_fo_gis_adjoint_sensitivity_thickness.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
    add_test(NAME ${testName}_GisAdjointSensitivityTwoParameters_DecompMesh
             COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
             -DTEST_NAME=GisAdjointSensitivityTwoParameters_DecompMesh 
             -DTEST_ARGS=input_fo_gis_analysis_two_params.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
  endif()
set_tests_properties(${testName}_GisRestartUnstructured_DecompMesh  PROPERTIES DEPENDS ${testName}_GisUnstructured_DecompMesh)
endif()
endif()
if(ALBANY_IFPACK2)
add_test(${testName}_Gis20km_Tpetra ${AlbanyT.exe} input_fo_gis20km_testT.yaml)
if (ALBANY_IOPX)
add_test(${testName}_GisUnstructuredPerformance_Tpetra ${AlbanyT.exe} input_fo_gis_unstruct_perfT.yaml)
set_tests_properties(${testName}_GisUnstructuredPerformance_Tpetra PROPERTIES DEPENDS ${testName}_GisPopulateMeshes)
add_test(${testName}_GisUnstructuredTpetra ${AlbanyT.exe} input_fo_gis_unstructT.yaml)
set_tests_properties(${testName}_GisUnstructuredTpetra PROPERTIES DEPENDS ${testName}_GisPopulateMeshes)
add_test(${testName}_GisAdjointSensitivityTpetra ${AlbanyT.exe} input_fo_gis_adjoint_sensitivityT.yaml)
add_test(${testName}_GisAdjointSensitivityBasalFrictionTpetra ${AlbanyT.exe} input_fo_gis_analysis_betaT.yaml)
add_test(${testName}_GisAdjointSensitivityStiffeningBasalFrictionTpetra ${AlbanyT.exe} input_fo_gis_analysis_stiffeningT.yaml)
add_test(${testName}_GisSensSMBwrtBetaTpetra ${AlbanyT.exe} input_fo_gis_beta_smbT.yaml)
  if (ALBANY_MESH_DEPENDS_ON_PARAMETERS)
    add_test(${testName}_GisAdjointSensitivityThicknessTpetra ${AlbanyT.exe} input_fo_gis_adjoint_sensitivity_thicknessT.yaml)
    add_test(${testName}_GisAdjointSensitivityThicknessMoveSurfHeightTpetra ${AlbanyT.exe} input_fo_gis_adjoint_sensitivity_thickness_adjustSurfHeightT.yaml)
    add_test(${testName}_GisAdjointSensitivityTwoParametersTpetra ${AlbanyT.exe} input_fo_gis_analysis_two_paramsT.yaml)
  endif()
  if (ALBANY_MESH_DEPENDS_ON_SOLUTION)
  add_test(${testName}_GisCoupledThicknessShapeOptTpetra ${AlbanyT.exe} input_fo_gis_coupled_shape_optT.yaml)
  else()
    add_test(${testName}_GisCoupledThicknessTpetra ${AlbanyT.exe} input_fo_gis_coupledT.yaml)
  endif()
else()
  add_test(NAME ${testName}_GisUnstructuredPerformanceTpetra_DecompMesh
           COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
             -DTEST_NAME=GisUnstructuredPerformanceTpetra_DecompMesh 
             -DTEST_ARGS=input_fo_gis_unstruct_perfT.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
  set_tests_properties(${testName}_GisUnstructuredPerformanceTpetra_DecompMesh PROPERTIES DEPENDS ${testName}_GisPopulateMeshes_DecompMesh)
  add_test(NAME ${testName}_GisUnstructuredTpetra_DecompMesh
           COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
             -DTEST_NAME=GisUnstructuredTpetra_DecompMesh 
             -DTEST_ARGS=input_fo_gis_unstructT.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
  set_tests_properties(${testName}_GisUnstructuredTpetra_DecompMesh PROPERTIES DEPENDS ${testName}_GisPopulateMeshes_DecompMesh)
  add_test(NAME ${testName}_GisAdjointSensitivityTpetra_DecompMesh
           COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
             -DTEST_NAME=GisAdjointSensitivityTpetra_DecompMesh 
             -DTEST_ARGS=input_fo_gis_adjoint_sensitivityT.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
  add_test(NAME ${testName}_GisAdjointSensitivityBasalFrictionTpetra_DecompMesh
           COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
             -DTEST_NAME=GisAdjointSensitivityBasalFrictionTpetra_DecompMesh 
             -DTEST_ARGS=input_fo_gis_analysis_betaT.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
  add_test(NAME ${testName}_GisAdjointSensitivityStiffeningBasalFrictionTpetra_DecompMesh
           COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
             -DTEST_NAME=GisAdjointSensitivityStiffeningBasalFrictionTpetra_DecompMesh 
             -DTEST_ARGS=input_fo_gis_analysis_stiffeningT.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
  add_test(NAME ${testName}_GisSensSMBwrtBetaTpetra_DecompMesh
           COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
             -DTEST_NAME=GisSensSMBwrtBetaTpetra_DecompMesh 
             -DTEST_ARGS=input_fo_gis_beta_smbT.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
  if (ALBANY_MESH_DEPENDS_ON_PARAMETERS)
    add_test(NAME ${testName}_GisAdjointSensitivityThicknessTpetra_DecompMesh
             COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
             -DTEST_NAME=GisAdjointSensitivityThicknessTpetra_DecompMesh 
             -DTEST_ARGS=input_fo_gis_adjoint_sensitivity_thicknessT.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
    add_test(NAME ${testName}_GisAdjointSensitivityThicknessMoveSurfHeightTpetra_DecompMesh
             COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
             -DTEST_NAME=GisAdjointSensitivityThicknessMoveSurfHeightTpetra_DecompMesh 
             -DTEST_ARGS=input_fo_gis_adjoint_sensitivity_thickness_adjustSurfHeightT.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
    add_test(NAME ${testName}_GisAdjointSensitivityTwoParametersTpetra_DecompMesh
             COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
             -DTEST_NAME=GisAdjointSensitivityTwoParametersTpetra_DecompMesh 
             -DTEST_ARGS=input_fo_gis_analysis_two_paramsT.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
  endif()
  if (ALBANY_MESH_DEPENDS_ON_SOLUTION)
    add_test(NAME ${testName}_GisCoupledThicknessShapeOptTpetra_DecompMesh
             COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
             -DTEST_NAME=GisCoupledThicknessShapeOptTpetra_DecompMesh 
             -DTEST_ARGS=input_fo_gis_coupled_shape_optT.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
  else()
    add_test(NAME ${testName}_GisCoupledThicknessTpetra_DecompMesh
             COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
             -DTEST_NAME=GisCoupledThicknessTpetra_DecompMesh 
             -DTEST_ARGS=input_fo_gis_coupledT.yaml -DMPIMNP=4
             -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
  endif()
endif(ALBANY_IOPX)
endif(ALBANY_IFPACK2)
