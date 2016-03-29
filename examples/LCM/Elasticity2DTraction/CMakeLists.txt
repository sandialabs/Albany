
# This test is not setup to run on the BGQ as it sits...
IF(NOT ALBANY_PARALLEL_ONLY)

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.xml COPYONLY)

# 2. Copy mesh files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hole.exo
               ${CMAKE_CURRENT_BINARY_DIR}/hole.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hole.exo.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/hole.exo.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hole.exo.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/hole.exo.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hole.exo.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/hole.exo.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hole.exo.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/hole.exo.4.3 COPYONLY)

# 3. Copy runtest.cmake from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake COPYONLY)

# 4. Copy the reference solution over
if (ALBANY_DTK) 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reference_hole_dtk.exo
               ${CMAKE_CURRENT_BINARY_DIR}/reference_hole_dtk.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest_dtk.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtest_dtk.cmake COPYONLY)
else()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reference_hole.exo
               ${CMAKE_CURRENT_BINARY_DIR}/reference_hole.exo COPYONLY)
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/exodiff_commands
               ${CMAKE_CURRENT_BINARY_DIR}/exodiff_commands COPYONLY)

# 4. Build the test executable to read the ncdump file
#add_executable(TestNcdumpValues TestNcdumpValues.C)

# 5. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 6. Run the "runtest.cmake" file in the CWD

if (ALBANY_EPETRA) 
if (ALBANY_DTK) 
add_test(NAME ${testName} COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
  -DTEST_NAME=${testName} -DMPIMNP=${MPIMNP} -DSEACAS_EPU=${SEACAS_EPU} -DSEACAS_EXODIFF=${SEACAS_EXODIFF} -P runtest_dtk.cmake)
else()
add_test(NAME ${testName} COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
  -DTEST_NAME=${testName} -DMPIMNP=${MPIMNP} -DSEACAS_EPU=${SEACAS_EPU} -DSEACAS_EXODIFF=${SEACAS_EXODIFF} -P runtest.cmake)
endif()
set_tests_properties(${testName}  PROPERTIES REQUIRED_FILES "${SEACAS_EPU};${SEACAS_EXODIFF}")
endif()

# 6i'. Run the "runtest.cmake" file in the CWD

if (ALBANY_IFPACK2)
if (ALBANY_DTK)
add_test(NAME ${testName}_Tpetra COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe};inputT.xml"
  -DTEST_NAME=${testName}_Tpetra -DMPIMNP=${MPIMNP} -DSEACAS_EPU=${SEACAS_EPU} -DSEACAS_EXODIFF=${SEACAS_EXODIFF} -DOUTFILE_SUFFIX=_tpetra
  -P runtest_dtk.cmake)
else() 
add_test(NAME ${testName}_Tpetra COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe};inputT.xml"
  -DTEST_NAME=${testName}_Tpetra -DMPIMNP=${MPIMNP} -DSEACAS_EPU=${SEACAS_EPU} -DSEACAS_EXODIFF=${SEACAS_EXODIFF} -DOUTFILE_SUFFIX=_tpetra
  -P runtest.cmake)
endif()
set_tests_properties(${testName}_Tpetra PROPERTIES REQUIRED_FILES "${SEACAS_EPU};${SEACAS_EXODIFF}")
endif()

ENDIF()
