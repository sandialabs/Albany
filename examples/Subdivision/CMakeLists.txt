##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# This test is currently not appropriate for the BGQ
IF(NOT ALBANY_PARALLEL_ONLY)

# Create a symlink to the Subdivision executable
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${Subdivision.exe} ${CMAKE_CURRENT_BINARY_DIR}/Subdivision)

# Create a symlink to exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy script file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_subdivision.py
               ${CMAKE_CURRENT_BINARY_DIR}/run_subdivision.py COPYONLY)

# Copy the input file to the test directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/one_tet.g
               ${CMAKE_CURRENT_BINARY_DIR}/one_tet.g COPYONLY)

# Copy the reference solution and exodiff files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/one_tet.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/one_tet.gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/one_tet.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/one_tet.exodiff COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#MESSAGE(SEACAS_EXODIFF "${SEACAS_EXODIFF}")

# Run the "runtest.py" file in the CWD
add_test(NAME ${testName}_one_tet COMMAND "python" "run_subdivision.py")
set_tests_properties(${testName}_one_tet  PROPERTIES REQUIRED_FILES "${SEACAS_EXODIFF}")

ENDIF()
