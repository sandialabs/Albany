//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanicsT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/mechanicsT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputElastAdaptT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputElastAdaptT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechNodeSizeFieldT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/mechNodeSizeFieldT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/smooth_tension_coarse_tet4.exo
               ${CMAKE_CURRENT_BINARY_DIR}/smooth_tension_coarse_tet4.exo COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable

if (ALBANY_IFPACK2 AND SEACAS_EPU AND SEACAS_EXODIFF)
add_test(${testName}_ElasticML ${AlbanyT.exe} inputElastAdaptT.yaml)
add_test(${testName} ${AlbanyT.exe} mechanicsT.yaml)

add_test(NAME ${testName}_NodeSizeField
         COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
         -DFILE_NAME=tet4_node_size_field_test_output_1 -DTEST_ARGS=mechNodeSizeFieldT.yaml -DMPIMNP=${MPIMNP}
         -DSEACAS_EPU=${SEACAS_EPU} -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
         -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)

endif()

