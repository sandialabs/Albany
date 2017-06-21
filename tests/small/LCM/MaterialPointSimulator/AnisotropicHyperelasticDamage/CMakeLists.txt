##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

if(SEACAS_EXODIFF)
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
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AHD.py
               ${CMAKE_CURRENT_BINARY_DIR}/AHD.py COPYONLY)

# Copy Input files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AHD-uniaxial.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/AHD-uniaxial.yaml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AHD-shear.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/AHD-shear.yaml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AHD-hydrostatic.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/AHD-hydrostatic.yaml COPYONLY)

# Copy the reference solution and exodiff files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AHD-uniaxial.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/AHD-uniaxial.gold.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AHD-uniaxial.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/AHD-uniaxial.exodiff COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AHD-shear.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/AHD-shear.gold.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AHD-shear.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/AHD-shear.exodiff COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AHD-hydrostatic.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/AHD-hydrostatic.gold.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AHD-hydrostatic.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/AHD-hydrostatic.exodiff COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Run the "runtest.py" file in the CWD
add_test(NAME ${testName} COMMAND "python" "AHD.py")
endif(SEACAS_EXODIFF)
