# 
# 1. Copy input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nlopt_input.xml ${CMAKE_CURRENT_BINARY_DIR}/nlopt_input.xml COPYONLY)
# 2. Copy mesh files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/modal.gen ${CMAKE_CURRENT_BINARY_DIR}/modal.gen COPYONLY)

# 3. Copy runtest.cmake from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake COPYONLY)

# 5. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 6. Run the "runtest.cmake" file in the CWD
add_test(NAME ${testName} COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
  -DTEST_NAME=${testName} -P runtest.cmake)

