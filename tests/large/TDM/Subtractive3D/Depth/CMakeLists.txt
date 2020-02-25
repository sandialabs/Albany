
# The root name for the tests in this directory is the directory name
get_filename_component(testNameRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/Input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Input_Parallel.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/Input_Parallel.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/LaserCenter.txt
               ${CMAKE_CURRENT_BINARY_DIR}/LaserCenter.txt COPYONLY)

#tests
add_test(TDM_${testNameRoot}_Serial ${SerialAlbany.exe} Input.yaml)
add_test(TDM_${testNameRoot}_Parallel ${Albany.exe} Input_Parallel.yaml)
set_tests_properties(TDM_${testNameRoot}_Serial PROPERTIES LABELS "TDM;Tpetra;Forward")
set_tests_properties(TDM_${testNameRoot}_Parallel PROPERTIES LABELS "TDM;Tpetra;Forward")
