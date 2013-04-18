##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AHD.xml
               ${CMAKE_CURRENT_BINARY_DIR}/AHD.xml COPYONLY)

# Copy runtest.py from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.py
               ${CMAKE_CURRENT_BINARY_DIR}/runtest.py COPYONLY)

# Create a symlink to the MPS
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${MPS.exe} ${CMAKE_CURRENT_BINARY_DIR}/MPS)

# Create a symlink to exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy the reference solution and exodiff file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Composite.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/Composite.gold.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Composite.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/Composite.exodiff COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#MESSAGE(SEACAS_EXODIFF "${SEACAS_EXODIFF}")

# Run the "runtest.py" file in the CWD
add_test(NAME ${testName} COMMAND "python" "runtest.py")
set_tests_properties(${testName}  PROPERTIES REQUIRED_FILES "${SEACAS_EXODIFF}")
