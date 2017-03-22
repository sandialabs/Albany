if (SEACAS_EPU AND SEACAS_EXODIFF)
# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTPS.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputTPS.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputLB.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputLB.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/3d_cgrid_small.e
               ${CMAKE_CURRENT_BINARY_DIR}/3d_cgrid_small.e COPYONLY)

# 4. Name the test with the directory name
get_filename_component(testDirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 5. Create the test with this name and standard executable
if (ALBANY_IFPACK2)
  set(testName ${testDirName})
  add_test(NAME LaplaceBeltrami:${testName}_Tpetra 
         COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
         -DTEST_NAME=${testName} -DTEST_ARGS= -DMPIMNP=${MPIMNP} 
         -DSEACAS_EPU=${SEACAS_EPU} -DSEACAS_EXODIFF=${SEACAS_EXODIFF} 
         -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)

  set(testName ${testDirName}_TPS)
  add_test(NAME LaplaceBeltrami:${testName}_Tpetra 
         COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
         -DTEST_NAME=${testName} -DTEST_ARGS=inputTPS.xml -DMPIMNP=${MPIMNP} 
         -DSEACAS_EPU=${SEACAS_EPU} -DSEACAS_EXODIFF=${SEACAS_EXODIFF} 
         -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)

  set(testName ${testDirName}_LB)
  add_test(NAME LaplaceBeltrami:${testName}_Tpetra 
         COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
         -DTEST_NAME=${testName} -DTEST_ARGS=inputLB.xml -DMPIMNP=${MPIMNP} 
         -DSEACAS_EPU=${SEACAS_EPU} -DSEACAS_EXODIFF=${SEACAS_EXODIFF} 
         -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
endif()
endif(SEACAS_EPU AND SEACAS_EXODIFF)
