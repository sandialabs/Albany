# Name the test with the directory name

get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${parentDirName}_${dirName})

#############################################
###  Test the column-coupling evaluators  ###
#############################################

set (testName ${testNameRoot}_Unstructured)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_column_coupling.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_column_coupling.yaml)
add_test(${testName} ${Albany.exe} input_column_coupling.yaml)
set_tests_properties(${testName} PROPERTIES
                     LABELS "LandIce;Tpetra;Forward")
