##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# SCOREC/meshes/notch ##################

# copy mesh files from current source dir to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notch.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/notch.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notch-serial0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/notch-serial0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notch0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/notch0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notch1.smb
               ${CMAKE_CURRENT_BINARY_DIR}/notch1.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notch2.smb
               ${CMAKE_CURRENT_BINARY_DIR}/notch2.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notch3.smb
               ${CMAKE_CURRENT_BINARY_DIR}/notch3.smb COPYONLY)
