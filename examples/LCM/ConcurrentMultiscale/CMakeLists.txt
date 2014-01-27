##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

#This test is not currently appropriate for the BGQ
IF(NOT ALBANY_PARALLEL_ONLY)

set(runtest.py ${CMAKE_CURRENT_SOURCE_DIR}/runtest.py)

# Create a symlink to the runtest.py script
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${runtest.py} ${CMAKE_CURRENT_BINARY_DIR}/runtest.py)

# Create a symlink to the MPS
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${MPS.exe} ${CMAKE_CURRENT_BINARY_DIR}/MPS)

# Create a symlink to exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy script file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Multiscale.py
               ${CMAKE_CURRENT_BINARY_DIR}/Multiscale.py COPYONLY)

# Copy Input files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Multiscale.xml
               ${CMAKE_CURRENT_BINARY_DIR}/Multiscale.xml COPYONLY)

# Copy the reference solution and exodiff files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Multiscale.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/Multiscale.gold.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Multiscale.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/Multiscale.exodiff COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Run the "runtest.py" file in the CWD
add_test(NAME ${testName} COMMAND "python" "Multiscale.py")
set_tests_properties(${testName}  PROPERTIES REQUIRED_FILES "${SEACAS_EXODIFF}")

ENDIF()
