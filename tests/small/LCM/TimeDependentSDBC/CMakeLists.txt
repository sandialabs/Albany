##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cube.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/finer_cube_sdbc.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/finer_cube_sdbc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/finer_cube_sdbc_file.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/finer_cube_sdbc_file.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/finer_cube_dbc.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/finer_cube_dbc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/finer_cube_dbc_file.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/finer_cube_dbc_file.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/time.dat
               ${CMAKE_CURRENT_BINARY_DIR}/time.dat COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/bc.dat
               ${CMAKE_CURRENT_BINARY_DIR}/bc.dat COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.g
               ${CMAKE_CURRENT_BINARY_DIR}/cube.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/finer_cube.g
               ${CMAKE_CURRENT_BINARY_DIR}/finer_cube.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/finer_cube.g.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/finer_cube.g.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/finer_cube.g.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/finer_cube.g.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/finer_cube.g.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/finer_cube.g.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/finer_cube.g.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/finer_cube.g.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/finer_cube.g.2.0
               ${CMAKE_CURRENT_BINARY_DIR}/finer_cube.g.2.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/finer_cube.g.2.1
               ${CMAKE_CURRENT_BINARY_DIR}/finer_cube.g.2.1 COPYONLY)

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyTPath} ${CMAKE_CURRENT_BINARY_DIR}/AlbanyT)
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
add_test(${testName} ${AlbanyT.exe} finer_cube_dbc.yaml)
add_test(${testName} ${AlbanyT.exe} finer_cube_dbc_file.yaml)
add_test(${testName} ${AlbanyT.exe} finer_cube_sdbc.yaml)
add_test(${testName} ${AlbanyT.exe} finer_cube_sdbc_file.yaml)
