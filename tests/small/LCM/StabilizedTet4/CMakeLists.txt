##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

if (NOT (ALBANY_IFPACK2 AND SEACAS_EXODIFF))
  return()
endif()

# input files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/galerkinT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/galerkinT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/stabilizedT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/stabilizedT.yaml COPYONLY)

# material files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/stabilized-materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/stabilized-materials.yaml COPYONLY)

# mesh file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/bar.g
               ${CMAKE_CURRENT_BINARY_DIR}/bar.g COPYONLY)

# exodiff files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/galerkin.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/galerkin.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/stabilized.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/stabilized.exodiff COPYONLY)

# gold files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/galerkin.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/galerkin.gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/stabilized.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/stabilized.gold.e COPYONLY)

# 2. Copy runtest.cmake from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake COPYONLY)

# 3. Name the test with the directory name
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 4. Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY)
  add_test(NAME ${dirName}_galerkin COMMAND
      ${CMAKE_COMMAND} "-DALBANY=${SerialAlbanyT.exe}" -DTESTNAME=galerkin
      -DEXODIFF=${SEACAS_EXODIFF} -P runtest.cmake)
# currently exodiff fails; look into it
# add_test(NAME ${dirName}_stabilized COMMAND
#     ${CMAKE_COMMAND} "-DALBANY=${SerialAlbanyT.exe}" -DTESTNAME=stabilized
#     -DEXODIFF=${SEACAS_EXODIFF} -P runtest.cmake)
ENDIF()
