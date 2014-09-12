##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# This test is not currently set up to run on the BGQ...
IF(NOT ALBANY_PARALLEL_ONLY)

# Create a symlink to the Partition executable
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${PartitionTest.exe} ${CMAKE_CURRENT_BINARY_DIR}/PartitionTest)

# Create a symlink to exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy script file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_partition.py
               ${CMAKE_CURRENT_BINARY_DIR}/run_partition.py COPYONLY)

# Copy the input file to the test directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.e
               ${CMAKE_CURRENT_BINARY_DIR}/input.e COPYONLY)

# Copy the reference solution and exodiff files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/partition.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/partition.gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/partition.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/partition.exodiff COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
add_test(NAME ${testName} COMMAND "python" "run_partition.py")
set_tests_properties(${testName}  PROPERTIES REQUIRED_FILES "${SEACAS_EXODIFF}")

ENDIF()
