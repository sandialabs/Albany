
# The root name for the tests in this directory is the directory name
get_filename_component(testNameRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Copy Input file from source to binary dir
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Input.yaml
#               ${CMAKE_CURRENT_BINARY_DIR}/Input.yaml COPYONLY)

#FIXME: add tests 
#add_test(${testName} ${Albany.exe} input.yaml)
#set_tests_properties(${testName} PROPERTIES LABELS "Basic;Tpetra;Forward")
