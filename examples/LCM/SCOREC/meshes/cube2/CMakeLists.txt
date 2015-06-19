##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# SCOREC/meshes/cube2 ##################

# copy mesh files from current source dir to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube2.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/cube2.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube2-serial0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cube2-serial0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube2-0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cube2-0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube2-1.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cube2-1.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube2-2.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cube2-2.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube2-3.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cube2-3.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube2-coarse-serial0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cube2-coarse-serial0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube2-coarse0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cube2-coarse0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube2-coarse1.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cube2-coarse1.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube2-coarse2.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cube2-coarse2.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube2-coarse3.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cube2-coarse3.smb COPYONLY)
