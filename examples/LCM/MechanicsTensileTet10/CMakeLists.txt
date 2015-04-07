##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/NotchedTensileTet10.xml
               ${CMAKE_CURRENT_BINARY_DIR}/NotchedTensileTet10.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched_tensile_tet10.g
               ${CMAKE_CURRENT_BINARY_DIR}/notched_tensile_tet10.g COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
IF(ALBANY_SEACAS)
  if (ALBANY_EPETRA) 
    add_test(${testName}_STK ${Albany.exe} NotchedTensileTet10.xml)
  endif()
if (ALBANY_MUELU_EXAMPLES)
  if (ALBANY_64BIT_INT)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/NotchedTensileTet10T_MueLu_64.xml
               ${CMAKE_CURRENT_BINARY_DIR}/NotchedTensileTet10T_MueLu_64.xml COPYONLY)
    add_test(${testName}_STK_Tpetra ${AlbanyT.exe} NotchedTensileTet10T_MueLu_64.xml)
  else ()
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/NotchedTensileTet10T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/NotchedTensileTet10T.xml COPYONLY)
    add_test(${testName}_STK_Tpetra ${AlbanyT.exe} NotchedTensileTet10T.xml)
  endif ()
endif()
ENDIF()
