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
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Neohookean-uniaxial.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/Neohookean-uniaxial.yaml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Neohookean-shear.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/Neohookean-shear.yaml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Neohookean-hydrostatic.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/Neohookean-hydrostatic.yaml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Neohookean-general.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/Neohookean-general.yaml COPYONLY)

# Copy the reference solution and exodiff files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Neohookean-uniaxial.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/Neohookean-uniaxial.gold.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Neohookean-uniaxial.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/Neohookean-uniaxial.exodiff COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Neohookean-shear.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/Neohookean-shear.gold.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Neohookean-shear.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/Neohookean-shear.exodiff COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Neohookean-hydrostatic.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/Neohookean-hydrostatic.gold.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Neohookean-hydrostatic.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/Neohookean-hydrostatic.exodiff COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Neohookean-general.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/Neohookean-general.gold.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Neohookean-general.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/Neohookean-general.exodiff COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Run the tests
#test 1 - uniaxial
SET(OUTFILE "Neohookean-uniaxial.exo")
SET(REF_FILE "Neohookean-uniaxial.gold.exo")
add_test(NAME ${testName}_uniaxial
         COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${MPS.exe}"
         -DTEST_NAME=Neohookean-uniaxial -DTEST_ARGS=--input=Neohookean-uniaxial.yaml -DMPIMNP=1
         -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
         -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
         -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${MPS.cmake})
#test 2 - shear
SET(OUTFILE "Neohookean-shear.exo")
SET(REF_FILE "Neohookean-shear.gold.exo")
add_test(NAME ${testName}_shear
         COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${MPS.exe}"
         -DTEST_NAME=Neohookean-shear -DTEST_ARGS=--input=Neohookean-shear.yaml -DMPIMNP=1
         -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
         -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
         -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${MPS.cmake})
#test 3 - hydrostatic
SET(OUTFILE "Neohookean-hydrostatic.exo")
SET(REF_FILE "Neohookean-hydrostatic.gold.exo")
add_test(NAME ${testName}_hydrostatic
         COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${MPS.exe}"
         -DTEST_NAME=Neohookean-hydrostatic -DTEST_ARGS=--input=Neohookean-hydrostatic.yaml -DMPIMNP=1
         -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
         -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
         -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${MPS.cmake})
#test 4 - general
SET(OUTFILE "Neohookean-general.exo")
SET(REF_FILE "Neohookean-general.gold.exo")
add_test(NAME ${testName}_general
         COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${MPS.exe}"
         -DTEST_NAME=Neohookean-general -DTEST_ARGS=--input=Neohookean-general.yaml -DMPIMNP=1
         -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
         -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
         -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${MPS.cmake}) 
endif(SEACAS_EXODIFF)
