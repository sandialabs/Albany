# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_with_source_dist_param_adjoint_sens_explicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_with_source_dist_param_adjoint_sens_explicit.yaml COPYONLY)

# Create the test with this name and standard executable
add_test(${testName}_Fwd_Sens_Implicit ${Albany.exe} input.yaml)
set_tests_properties(${testName}_Fwd_Sens_Implicit PROPERTIES LABELS "Demo;Tpetra;Forward;Serial")
add_test(${testName}_with_Source_Dist_Param_Explicit ${SerialAlbany.exe} input_with_source_dist_param_adjoint_sens_explicit.yaml)
set_tests_properties(${testName}_with_Source_Dist_Param_Explicit PROPERTIES LABELS
                                       	"Demo;Tpetra;Adjoint;Serial")
