##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# SCOREC/meshes/two_bars ##################

# copy mesh files from current source dir to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/two_bars.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/two_bars.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/two_bars0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/two_bars0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/two_bars1.smb
               ${CMAKE_CURRENT_BINARY_DIR}/two_bars1.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/two_bars2.smb
               ${CMAKE_CURRENT_BINARY_DIR}/two_bars2.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/two_bars3.smb
               ${CMAKE_CURRENT_BINARY_DIR}/two_bars3.smb COPYONLY)
