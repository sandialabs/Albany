# Name the test with the directory name

get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${parentDirName}_${dirName})

#############################################
###  Test the column-coupling evaluators  ###
#############################################

set (RESIDUAL_SIDE "basalside")
set (DIRICHLET_SIDE "top")
set (testName ${testNameRoot}_bed)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_column_coupling.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_column_coupling_bed.yaml)
add_test(${testName} ${Albany.exe} input_column_coupling_bed.yaml)
set_tests_properties(${testName} PROPERTIES
                     LABELS "LandIce;Tpetra;Forward")

set (RESIDUAL_SIDE "upperside")
set (DIRICHLET_SIDE "bottom")
set (testName ${testNameRoot}_surface)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_column_coupling.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_column_coupling_surface.yaml)
add_test(${testName} ${Albany.exe} input_column_coupling_surface.yaml)
set_tests_properties(${testName} PROPERTIES
                     LABELS "LandIce;Tpetra;Forward")
