##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# SCOREC/meshes/cylinder ##################

# copy mesh files from current source dir to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cylinder.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/cylinder.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cylinder-serial0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cylinder-serial0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cylinder0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cylinder0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cylinder1.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cylinder1.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cylinder2.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cylinder2.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cylinder3.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cylinder3.smb COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2ndOrderCyl.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/2ndOrderCyl.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2ndOrderCyl0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/2ndOrderCyl0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2ndOrderCyl_p0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/2ndOrderCyl_p0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2ndOrderCyl_p1.smb
               ${CMAKE_CURRENT_BINARY_DIR}/2ndOrderCyl_p1.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2ndOrderCyl_p2.smb
               ${CMAKE_CURRENT_BINARY_DIR}/2ndOrderCyl_p2.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2ndOrderCyl_p3.smb
               ${CMAKE_CURRENT_BINARY_DIR}/2ndOrderCyl_p3.smb COPYONLY)
