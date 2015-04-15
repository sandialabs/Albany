##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# AMP/meshes/ ##################

# copy mesh files from current source dir to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/twoLayers.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/twoLayers.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/twoLayersSerial0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/twoLayersSerial0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/twoLayers0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/twoLayers0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/twoLayers1.smb
               ${CMAKE_CURRENT_BINARY_DIR}/twoLayers1.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/twoLayers2.smb
               ${CMAKE_CURRENT_BINARY_DIR}/twoLayers2.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/twoLayers3.smb
               ${CMAKE_CURRENT_BINARY_DIR}/twoLayers3.smb COPYONLY)
