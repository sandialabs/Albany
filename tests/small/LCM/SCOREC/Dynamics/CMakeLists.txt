##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dynamicsT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/dynamicsT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dynamicsAdaptT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/dynamicsAdaptT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/serialDynamicsAdaptT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/serialDynamicsAdaptT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/elastic.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/elastic.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/plane.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/plane.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/plane0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/plane0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tri_plane0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/tri_plane0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tri_plane1.smb
               ${CMAKE_CURRENT_BINARY_DIR}/tri_plane1.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tri_plane2.smb
               ${CMAKE_CURRENT_BINARY_DIR}/tri_plane2.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tri_plane3.smb
               ${CMAKE_CURRENT_BINARY_DIR}/tri_plane3.smb COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
if (ALBANY_IFPACK2)
  add_test(${testName}_SCOREC_Tpetra ${AlbanyT.exe} dynamicsT.yaml)
  add_test(${testName}_SCOREC_Adapt_Tpetra ${AlbanyT.exe} dynamicsAdaptT.yaml)
endif()
