##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/NotchedTensileTet10.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/NotchedTensileTet10.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched_tensile_tet10.g
               ${CMAKE_CURRENT_BINARY_DIR}/notched_tensile_tet10.g COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
IF(ALBANY_SEACAS)
  if (ALBANY_EPETRA) 
    add_test(${testName}_STK ${Albany.exe} NotchedTensileTet10.yaml)
  endif()
  if (ALBANY_MUELU_EXAMPLES)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/NotchedTensileTet10T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/NotchedTensileTet10T.yaml COPYONLY)
    add_test(${testName}_STK ${AlbanyT.exe} NotchedTensileTet10T.yaml)
  endif()
ENDIF()
