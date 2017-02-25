##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# SCOREC/meshes/necking ##################

# copy mesh files from current source dir to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/necking.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/necking.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/necking-serial0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/necking-serial0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/necking0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/necking0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/necking1.smb
               ${CMAKE_CURRENT_BINARY_DIR}/necking1.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/necking2.smb
               ${CMAKE_CURRENT_BINARY_DIR}/necking2.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/necking3.smb
               ${CMAKE_CURRENT_BINARY_DIR}/necking3.smb COPYONLY)
