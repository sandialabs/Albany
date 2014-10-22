#*****************************************************************//
#    Albany 2.0:  Copyright 2012 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//


# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputEdgeDBCT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputEdgeDBCT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputVtxDBCT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputVtxDBCT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputRenameT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputRenameT.xml COPYONLY)

# 2. Name the test with the directory name, prepended by SCOREC_
get_filename_component(testRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(testName "SCOREC_${testRoot}")

# 3. Create the test with this name and standard executable
IF(ENABLE_MESHMODEL)
  IF(ALBANY_IFPACK2)
    add_test(
      NAME ${testName}_Traction_Tpetra
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND ${AlbanyT.exe} inputT.xml)
    add_test(
      NAME ${testName}_EdgeDBC_Tpetra
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND ${AlbanyT.exe} inputEdgeDBCT.xml)
    add_test(
      NAME ${testName}_VtxDBC_Tpetra
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND ${AlbanyT.exe} inputVtxDBCT.xml)
    add_test(
      NAME ${testName}_Rename_Tpetra
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND ${AlbanyT.exe} inputRenameT.xml)
  ENDIF()
ENDIF()
