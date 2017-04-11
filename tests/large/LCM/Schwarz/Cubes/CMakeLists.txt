##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.e
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1.e
               ${CMAKE_CURRENT_BINARY_DIR}/cube1.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-single.e
               ${CMAKE_CURRENT_BINARY_DIR}/cube-single.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials0.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials0.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials1.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials1.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials-single.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials-single.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cube1.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cubes.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cubes.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-single.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cube-single.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtestT.py
               ${CMAKE_CURRENT_BINARY_DIR}/runtestT.py COPYONLY)

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyTPath} ${CMAKE_CURRENT_BINARY_DIR}/AlbanyT)
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
add_test(NAME Schwarz_${testName} COMMAND "python" "runtestT.py")

IF(ALBANY_DTK)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cubes_matrix-free.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/cubes_matrix-free.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtestT_matrix-free.py
    ${CMAKE_CURRENT_BINARY_DIR}/runtestT_matrix-free.py COPYONLY)
  add_test(NAME Schwarz_${testName}_MatrixFree COMMAND "python" "runtestT_matrix-free.py")
ENDIF()
