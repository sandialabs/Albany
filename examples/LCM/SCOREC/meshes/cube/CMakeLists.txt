##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# SCOREC/meshes/cube ##################

# copy mesh files from current source dir to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/cube.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-serial0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cube-serial0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cube1.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube2.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cube2.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube3.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cube3.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-quad.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/cube-quad.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-quad-serial0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cube-quad-serial0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-quad0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cube-quad0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-quad1.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cube-quad1.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-quad2.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cube-quad2.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-quad3.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cube-quad3.smb COPYONLY)
