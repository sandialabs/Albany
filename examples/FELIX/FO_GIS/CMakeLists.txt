# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis20km_ascii.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis20km_ascii.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis20km.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis20km.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis20km_refine.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis20km_refine.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis10km_ascii.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis10km_ascii.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis10km.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis10km.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis5km_ascii.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis5km_ascii.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis5km.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis5km.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis2km_ascii.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis2km_ascii.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis2km.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis2km.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_unstruct.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_unstruct.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_coupled.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_coupled.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_coupled_shape_opt.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_coupled_shape_opt.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_manifold.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_manifold.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_unstruct_restart.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_unstruct_restart.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_adjoint_sensitivity.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_adjoint_sensitivity.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_wedge_adjoint_sensitivity.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_wedge_adjoint_sensitivity.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_adjoint_sensitivity_beta.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_adjoint_sensitivity_beta.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_adjoint_sensitivity_thickness.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_adjoint_sensitivity_thickness.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_beta.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_beta.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_thickness.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_thickness.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_analysis_two_params.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_analysis_two_params.xml COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#3. Tests 
#  First Test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis20km_test.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis20km_test.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis20km_testT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis20km_testT.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_Gis20km ${Albany.exe} input_fo_gis20km_test.xml)
add_test(${testName}_GisUnstructured ${Albany.exe} input_fo_gis_unstruct.xml)
if (ALBANY_MESH_DEPENDS_ON_SOLUTION) 
  add_test(${testName}_GisCoupledThicknessShapeOpt ${Albany.exe} input_fo_gis_coupled_shape_opt.xml)
else()
  add_test(${testName}_GisCoupledThickness ${Albany.exe} input_fo_gis_coupled.xml)
endif()
add_test(${testName}_GisManifold ${Albany.exe} input_fo_gis_manifold.xml)
add_test(${testName}_GisRestartUnstructured ${Albany.exe} input_fo_gis_unstruct_restart.xml)
add_test(${testName}_GisAdjointSensitivity ${Albany.exe} input_fo_gis_adjoint_sensitivity.xml)
add_test(${testName}_GisWedgeAdjointSensitivity ${Albany.exe} input_fo_gis_wedge_adjoint_sensitivity.xml)
if (ALBANY_EPETRA)
add_test(${testName}_GisAdjointSensitivityBasalFriction ${Albany.exe} input_fo_gis_adjoint_sensitivity_beta.xml)
  if (ALBANY_MESH_DEPENDS_ON_PARAMETERS)
    add_test(${testName}_GisAdjointSensitivityThickness ${Albany.exe} input_fo_gis_adjoint_sensitivity_thickness.xml)
    add_test(${testName}_GisAdjointSensitivityTwoParameters ${Albany.exe} input_fo_gis_analysis_two_params.xml)
  endif()
endif()
set_tests_properties(${testName}_GisRestartUnstructured  PROPERTIES DEPENDS ${testName}_GisUnstructured) 
set_tests_properties(${testName}_Gis20km  PROPERTIES LABELS CUDA_TEST) 
endif() 
if(ALBANY_IFPACK2)
add_test(${testName}_Gis20km_Tpetra ${AlbanyT.exe} input_fo_gis20km_testT.xml)
endif(ALBANY_IFPACK2)

#IK, 10/29/13: Convert to tpetra!
IF(ALBANY_STK_PERCEPT AND ALBANY_EPETRA)
add_test(${testName}_Gis20km_refine ${Albany8.exe} input_fo_gis20km_refine.xml)
ENDIF()

#salloc -n 4 -n 4 -p pbatch
#mpirun -np 4 ../../../build/src/Albany input_fo_gis20km.xml
