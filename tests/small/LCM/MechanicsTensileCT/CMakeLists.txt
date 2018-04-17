##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/NotchedTensileCT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/NotchedTensileCT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched_tensile.g
               ${CMAKE_CURRENT_BINARY_DIR}/notched_tensile.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched_tensile.g.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/notched_tensile.g.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched_tensile.g.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/notched_tensile.g.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched_tensile.g.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/notched_tensile.g.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched_tensile.g.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/notched_tensile.g.4.3 COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
IF (ALBANY_MUELU_EXAMPLES)
  add_test(${testName} ${AlbanyT.exe} NotchedTensileCT.yaml)
ENDIF()
