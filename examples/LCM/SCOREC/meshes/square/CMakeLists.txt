##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# SCOREC/meshes/square ##################

# copy mesh files from current source dir to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/square.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square-serial0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/square-serial0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/square0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square1.smb
               ${CMAKE_CURRENT_BINARY_DIR}/square1.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square2.smb
               ${CMAKE_CURRENT_BINARY_DIR}/square2.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square3.smb
               ${CMAKE_CURRENT_BINARY_DIR}/square3.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square-quad-serial0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/square-quad-serial0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square-quad0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/square-quad0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square-quad1.smb
               ${CMAKE_CURRENT_BINARY_DIR}/square-quad1.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square-quad2.smb
               ${CMAKE_CURRENT_BINARY_DIR}/square-quad2.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square-quad3.smb
               ${CMAKE_CURRENT_BINARY_DIR}/square-quad3.smb COPYONLY)
