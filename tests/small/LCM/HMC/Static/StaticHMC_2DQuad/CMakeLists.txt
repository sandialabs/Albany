# Name the test with the directory name
get_filename_component(testDirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(testName ${testDirName})

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${testName}.gen  ${CMAKE_CURRENT_BINARY_DIR}/${testName}.gen COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${testName}.gen.4.0  ${CMAKE_CURRENT_BINARY_DIR}/${testName}.gen.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${testName}.gen.4.1  ${CMAKE_CURRENT_BINARY_DIR}/${testName}.gen.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${testName}.gen.4.2  ${CMAKE_CURRENT_BINARY_DIR}/${testName}.gen.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${testName}.gen.4.3  ${CMAKE_CURRENT_BINARY_DIR}/${testName}.gen.4.3 COPYONLY)

# Create the test with this name and standard executable
add_test(NAME HMC:${testName} 
         COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
         -DTEST_NAME=${testName} -DTEST_ARGS= -DMPIMNP=${MPIMNP} 
         -DSEACAS_EPU=${SEACAS_EPU} -DSEACAS_EXODIFF=${SEACAS_EXODIFF} 
         -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
set_tests_properties(HMC:${testName}  PROPERTIES REQUIRED_FILES "${SEACAS_EPU};${SEACAS_EXODIFF}")
