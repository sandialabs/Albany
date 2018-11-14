# Name the test with the directory name
get_filename_component(testDirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(testName ${testDirName})

# Copy Input file from source to binary dir
IF (ALBANY_EPETRA) 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3D.yaml ${CMAKE_CURRENT_BINARY_DIR}/mitchell3D.yaml COPYONLY)
ENDIF() 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3DT.yaml ${CMAKE_CURRENT_BINARY_DIR}/mitchell3DT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3D.gen ${CMAKE_CURRENT_BINARY_DIR}/mitchell3D.gen COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3D.gen.4.0 ${CMAKE_CURRENT_BINARY_DIR}/mitchell3D.gen.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3D.gen.4.1 ${CMAKE_CURRENT_BINARY_DIR}/mitchell3D.gen.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3D.gen.4.2 ${CMAKE_CURRENT_BINARY_DIR}/mitchell3D.gen.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3D.gen.4.3 ${CMAKE_CURRENT_BINARY_DIR}/mitchell3D.gen.4.3 COPYONLY)

IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # using Intel C++
  SET(REF_FILE "RegHeaviside_3D.ref_intel.exo")
ELSE()
  SET(REF_FILE "RegHeaviside_3D.ref.exo")
ENDIF()

# Create the test with this name and standard executable
IF (SEACAS_EPU AND SEACAS_EXODIFF)
IF (ALBANY_EPETRA) 
SET(OUTFILE "physics_0_mitchell3D.exo")
add_test(NAME ATO:${testName} 
         COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
         -DTEST_NAME=${testName} -DTEST_ARGS=mitchell3D.yaml -DMPIMNP=${MPIMNP} 
         -DSEACAS_EPU=${SEACAS_EPU} -DSEACAS_EXODIFF=${SEACAS_EXODIFF} 
         -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
         -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
ENDIF() 
SET(OUTFILE "physics_0_mitchell3DT.exo")
add_test(NAME ATOT:${testName} 
         COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
         -DTEST_NAME=${testName} -DTEST_ARGS=mitchell3DT.yaml -DMPIMNP=${MPIMNP} 
         -DSEACAS_EPU=${SEACAS_EPU} -DSEACAS_EXODIFF=${SEACAS_EXODIFF} 
         -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
         -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
ENDIF()
