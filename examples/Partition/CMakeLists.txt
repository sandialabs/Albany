

# 1. Copy the input file to the test directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.e
               ${CMAKE_CURRENT_BINARY_DIR}/input.e COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
add_test(${testName} ${PartitionTest.exe})

