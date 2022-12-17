# Name the test with the directory name
get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testName ${parentDirName}_${dirName})

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_with_source_scalar_param_fwd_sens_implicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_with_source_scalar_param_fwd_sens_implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_with_source_scalar_param_adjoint_sens_implicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_with_source_scalar_param_adjoint_sens_implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_with_source_scalar_param_fwd_sens_explicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_with_source_scalar_param_fwd_sens_explicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_with_source_dist_param_adjoint_sens_implicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_with_source_dist_param_adjoint_sens_implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_with_source_dist_param_adjoint_sens_explicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_with_source_dist_param_adjoint_sens_explicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/compute_errs.m
               ${CMAKE_CURRENT_BINARY_DIR}/compute_errs.m COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/compute_errs_no_movie2avi.m
               ${CMAKE_CURRENT_BINARY_DIR}/compute_errs_no_movie2avi.m COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/compute_errs_source_no_movie2avi.m
               ${CMAKE_CURRENT_BINARY_DIR}/compute_errs_source_no_movie2avi.m COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/plot_exact_soln.m
               ${CMAKE_CURRENT_BINARY_DIR}/plot_exact_soln.m COPYONLY)

# Create the test with this name and standard executable
add_test(${testName} ${Albany.exe} input.yaml)
set_tests_properties(${testName} PROPERTIES LABELS
                                            "Demo;Tpetra;Forward;Serial")
add_test(${testName}_with_Source_Scalar_Param_Fwd_Sens_Implicit ${Albany.exe} input_with_source_scalar_param_fwd_sens_implicit.yaml)
set_tests_properties(${testName}_with_Source_Scalar_Param_Fwd_Sens_Implicit PROPERTIES LABELS
                                            "Demo;Tpetra;Forward;Serial")
add_test(${testName}_with_Source_Scalar_Param_Adjoint_Sens_Implicit ${Albany.exe} input_with_source_scalar_param_adjoint_sens_implicit.yaml)
set_tests_properties(${testName}_with_Source_Scalar_Param_Adjoint_Sens_Implicit PROPERTIES LABELS
	"Demo;Tpetra;Adjoint;Serial")
add_test(${testName}_with_Source_Dist_Param_Adjoint_Sens_Implicit ${SerialAlbany.exe} input_with_source_dist_param_adjoint_sens_implicit.yaml)
set_tests_properties(${testName}_with_Source_Dist_Param_Adjoint_Sens_Implicit PROPERTIES LABELS
                                       	"Demo;Tpetra;Adjoint;Serial")
add_test(${testName}_with_Source_Dist_Param_Adjoint_Sens_Explicit ${SerialAlbany.exe} input_with_source_dist_param_adjoint_sens_explicit.yaml)
set_tests_properties(${testName}_with_Source_Dist_Param_Adjoint_Sens_Explicit PROPERTIES LABELS
                                       	"Demo;Tpetra;Adjoint;Serial")
