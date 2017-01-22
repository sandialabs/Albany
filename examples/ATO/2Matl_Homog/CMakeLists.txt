# Name the test with the directory name
get_filename_component(testDirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(testName ${testDirName})

# 
# 1. Copy input file from source to binary dir
IF (ALBANY_EPETRA) 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3D.xml ${CMAKE_CURRENT_BINARY_DIR}/mitchell3D.xml COPYONLY)
ENDIF() 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3DT.xml ${CMAKE_CURRENT_BINARY_DIR}/mitchell3DT.xml COPYONLY)
# 2. Copy mesh files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/padded.gen     ${CMAKE_CURRENT_BINARY_DIR}/padded.gen COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/padded.gen.4.0 ${CMAKE_CURRENT_BINARY_DIR}/padded.gen.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/padded.gen.4.1 ${CMAKE_CURRENT_BINARY_DIR}/padded.gen.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/padded.gen.4.2 ${CMAKE_CURRENT_BINARY_DIR}/padded.gen.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/padded.gen.4.3 ${CMAKE_CURRENT_BINARY_DIR}/padded.gen.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.gen     ${CMAKE_CURRENT_BINARY_DIR}/cube.gen COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.gen.4.0 ${CMAKE_CURRENT_BINARY_DIR}/cube.gen.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.gen.4.1 ${CMAKE_CURRENT_BINARY_DIR}/cube.gen.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.gen.4.2 ${CMAKE_CURRENT_BINARY_DIR}/cube.gen.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.gen.4.3 ${CMAKE_CURRENT_BINARY_DIR}/cube.gen.4.3 COPYONLY)

# Create the test with this name and standard executable
IF (ALBANY_EPETRA) 
add_test(NAME ATO:${testName} 
         COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
         -DTEST_NAME=${testName} -DTEST_ARGS=mitchell3D.xml -DMPIMNP=${MPIMNP} 
         -DSEACAS_EPU=${SEACAS_EPU} -DSEACAS_EXODIFF=${SEACAS_EXODIFF} 
         -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
set_tests_properties(ATO:${testName}  PROPERTIES REQUIRED_FILES "${SEACAS_EPU};${SEACAS_EXODIFF}")
ENDIF() 
#IKT, FIXME: uncomment the following once this test passes 
#add_test(NAME ATOT:${testName} 
#         COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
#         -DTEST_NAME=${testName} -DTEST_ARGS=mitchell3DT.xml -DMPIMNP=${MPIMNP} 
#         -DSEACAS_EPU=${SEACAS_EPU} -DSEACAS_EXODIFF=${SEACAS_EXODIFF} 
#         -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtestT.cmake)
#set_tests_properties(ATOT:${testName}  PROPERTIES REQUIRED_FILES "${SEACAS_EPU};${SEACAS_EXODIFF}")
