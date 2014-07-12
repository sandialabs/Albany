##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# SCOREC ##################

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/meshes/cube/cube.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/meshes/cube/cube.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/meshes/cube/cube-serial0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/meshes/cube/cube-serial0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/meshes/cube/cube0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/meshes/cube/cube0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/meshes/cube/cube1.smb
               ${CMAKE_CURRENT_BINARY_DIR}/meshes/cube/cube1.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/meshes/cube/cube2.smb
               ${CMAKE_CURRENT_BINARY_DIR}/meshes/cube/cube2.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/meshes/cube/cube3.smb
               ${CMAKE_CURRENT_BINARY_DIR}/meshes/cube/cube3.smb COPYONLY)

add_subdirectory(Elasticity)
