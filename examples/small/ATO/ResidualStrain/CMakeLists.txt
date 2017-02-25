# 
# 1. Copy input file from source to binary dir
IF (ALBANY_EPETRA) 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.xml ${CMAKE_CURRENT_BINARY_DIR}/mitchell.xml COPYONLY)
ENDIF() 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchellT.xml ${CMAKE_CURRENT_BINARY_DIR}/mitchellT.xml COPYONLY)
# 2. Copy mesh files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen     ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen COPYONLY)
# 3. Copy runtest.cmake from source to binary dir
IF (ALBANY_EPETRA) 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake COPYONLY)
ENDIF() 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtestT.cmake ${CMAKE_CURRENT_BINARY_DIR}/runtestT.cmake COPYONLY)
# 5. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
IF (ALBANY_EPETRA) 
# 6. Run the "runtest.cmake" file in the CWD
add_test(NAME ATO:${testName} COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}" -DTEST_NAME=${testName} -P runtest.cmake)
ENDIF() 
add_test(NAME ATOT:${testName} COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}" -DTEST_NAME=${testName} -P runtestT.cmake)
