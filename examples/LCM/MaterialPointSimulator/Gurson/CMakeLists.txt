##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

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
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Gurson.py
               ${CMAKE_CURRENT_BINARY_DIR}/Gurson.py COPYONLY)

# Copy Input files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Gurson-uniaxial.xml
               ${CMAKE_CURRENT_BINARY_DIR}/Gurson-uniaxial.xml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Gurson-shear.xml
               ${CMAKE_CURRENT_BINARY_DIR}/Gurson-shear.xml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Gurson-hydrostatic.xml
               ${CMAKE_CURRENT_BINARY_DIR}/Gurson-hydrostatic.xml COPYONLY)

# Copy the reference solution and exodiff files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Gurson-uniaxial.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/Gurson-uniaxial.gold.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Gurson-uniaxial.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/Gurson-uniaxial.exodiff COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Gurson-shear.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/Gurson-shear.gold.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Gurson-shear.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/Gurson-shear.exodiff COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Gurson-hydrostatic.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/Gurson-hydrostatic.gold.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Gurson-hydrostatic.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/Gurson-hydrostatic.exodiff COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#MESSAGE(SEACAS_EXODIFF "${SEACAS_EXODIFF}")

# Run the "runtest.py" file in the CWD
add_test(NAME ${testName} COMMAND "python" "Gurson.py")
set_tests_properties(${testName}  PROPERTIES REQUIRED_FILES "${SEACAS_EXODIFF}")
