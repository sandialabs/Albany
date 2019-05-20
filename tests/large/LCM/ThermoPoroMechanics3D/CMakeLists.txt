
# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
# Comment out the Epetra test for now. This test generates FPEs on mpi 4 rank runs (not serial) - some closer examination
# needs to be done here to debug this. Please search for "GAH" in the LCM/evaluators directory to get more context.
#if (ALBANY_EPETRA)
#add_test(${testName} ${Albany.exe})
#endif()
if (ALBANY_IFPACK2)
add_test(${testName} ${Albany.exe} inputT.yaml)
set_tests_properties(${testName} PROPERTIES LABELS "LCM;Tpetra;Forward")
endif()

