# Name the test with the directory name
get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testName ${parentDirName}_${dirName})

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_scalar_param_fwd_sens_implicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_scalar_param_fwd_sens_implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_scalar_param_adjoint_sens_explicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_scalar_param_adjoint_sens_explicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_with_source_dist_param_adjoint_sens_explicit_consistentM.yaml
	       ${CMAKE_CURRENT_BINARY_DIR}/input_with_source_dist_param_adjoint_sens_explicit_consistentM.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_with_source_dist_param_adjoint_sens_explicit_lumpedM.yaml
     	       ${CMAKE_CURRENT_BINARY_DIR}/input_with_source_dist_param_adjoint_sens_explicit_lumpedM.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_with_source_dist_param_adjoint_sens_implicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_with_source_dist_param_adjoint_sens_implicit.yaml COPYONLY)

# Create the tests

add_test(${testName}_with_Source_Dist_Param_Adjoint_Sens_Explicit_ConsistentM
         ${SerialAlbany.exe} input_with_source_dist_param_adjoint_sens_explicit_consistentM.yaml)
set_tests_properties(${testName}_with_Source_Dist_Param_Adjoint_Sens_Explicit_ConsistentM
                     PROPERTIES LABELS "Demo;Adjoint;Serial")

add_test(${testName}_with_Source_Dist_Param_Adjoint_Sens_Explicit_LumpedM
         ${SerialAlbany.exe} input_with_source_dist_param_adjoint_sens_explicit_lumpedM.yaml)
set_tests_properties(${testName}_with_Source_Dist_Param_Adjoint_Sens_Explicit_LumpedM
                     PROPERTIES LABELS "Demo;Adjoint;Serial")

add_test(${testName}_with_Source_Dist_Param_Adjoint_Implicit
         ${SerialAlbany.exe} input_with_source_dist_param_adjoint_sens_implicit.yaml)
set_tests_properties(${testName}_with_Source_Dist_Param_Adjoint_Implicit PROPERTIES LABELS
	                                "Demo;Forward;Serial")

# These tests are more expensive, so enable only if user wants it (ON by default)
if (ALBANY_LARGE_TESTS)
  add_test(${testName}_Scalar_Param_Fwd_Sens_Implicit
           ${Albany.exe} input_scalar_param_fwd_sens_implicit.yaml)
  set_tests_properties(${testName}_Scalar_Param_Fwd_Sens_Implicit
                       PROPERTIES LABELS "Demo;Forward;Serial")

  add_test(${testName}_Scalar_Param_Adjoint_Sens_Explicit
           ${Albany.exe} input_scalar_param_adjoint_sens_explicit.yaml)
  set_tests_properties(${testName}_Scalar_Param_Adjoint_Sens_Explicit
                       PROPERTIES LABELS "Demo;Adjoint;Serial")
endif()
