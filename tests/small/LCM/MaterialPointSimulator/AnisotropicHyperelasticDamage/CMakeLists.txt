##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

if(SEACAS_EXODIFF)
# Create a symlink to the runtest.py script
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${runtest.py} ${CMAKE_CURRENT_BINARY_DIR}/runtest.py)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${MPS.cmake} ${CMAKE_CURRENT_BINARY_DIR}/MPS.cmake)

# Create a symlink to the MPS
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${MPS.exe} ${CMAKE_CURRENT_BINARY_DIR}/MPS)

# Create a symlink to exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

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

# Run the tests
#test 1 - uniaxial
SET(OUTFILE "AHD-uniaxial.exo")
SET(REF_FILE "AHD-uniaxial.gold.exo")
add_test(NAME ${testName}_uniaxial
         COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${MPS.exe}"
         -DTEST_NAME=AHD-uniaxial -DTEST_ARGS=--input=AHD-uniaxial.yaml -DMPIMNP=1
         -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
         -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
         -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${MPS.cmake})
#test 2 - shear
SET(OUTFILE "AHD-shear.exo")
SET(REF_FILE "AHD-shear.gold.exo")
add_test(NAME ${testName}_shear
         COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${MPS.exe}"
         -DTEST_NAME=AHD-shear -DTEST_ARGS=--input=AHD-shear.yaml -DMPIMNP=1
         -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
         -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
         -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${MPS.cmake})
#test 3 - hydrostatic
SET(OUTFILE "AHD-hydrostatic.exo")
SET(REF_FILE "AHD-hydrostatic.gold.exo")
add_test(NAME ${testName}_hydrostatic
         COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${MPS.exe}"
         -DTEST_NAME=AHD-hydrostatic -DTEST_ARGS=--input=AHD-hydrostatic.yaml -DMPIMNP=1
         -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
         -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
         -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${MPS.cmake})

endif(SEACAS_EXODIFF)
