##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

if (ALBANY_IFPACK2 AND SEACAS_EXODIFF)
# Create a symlink to exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
# Create a symlink to AlbanyT
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyTPath} ${CMAKE_CURRENT_BINARY_DIR}/AlbanyT)

# input files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/EqConcentrationBC.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/EqConcentrationBC.yaml COPYONLY)

# material files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SimpleMaterials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/SimpleMaterials.yaml COPYONLY)

# exodiff files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/EqConcentrationBC.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/EqConcentrationBC.exodiff COPYONLY)

# gold files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/EqConcentrationBC.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/EqConcentrationBC.gold.e COPYONLY)

# python runtest file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtestT.py
               ${CMAKE_CURRENT_BINARY_DIR}/runtestT.py COPYONLY)

# python verification file for plotting in test directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/plotEqConcentration.py
               ${CMAKE_CURRENT_BINARY_DIR}/plotEqConcentration.py COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY)
  add_test(NAME ${testName} COMMAND "python" "runtestT.py")
ENDIF()
endif()
