#*****************************************************************//
#    Albany 2.0:  Copyright 2012 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//


# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MeshModel/inputT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MeshModel/inputT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MeshModel/inputEdgeDBCT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MeshModel/inputEdgeDBCT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MeshModel/inputVtxDBCT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MeshModel/inputVtxDBCT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MeshModel/inputRenameT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MeshModel/inputRenameT.xml COPYONLY)

# 2. Name the test with the directory name, prepended by SCOREC_
get_filename_component(testRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(testName "SCOREC_${testRoot}")

# 3. Create the test with this name and standard executable
IF(ENABLE_MESHMODEL)
  if (ALBANY_IFPACK2)
    add_test(
      NAME ${testName}_Mmodel_Traction_Tpetra
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/MeshModel
      COMMAND ${AlbanyT.exe} inputT.xml)
    add_test(
      NAME ${testName}_Mmodel_EdgeDBC_Tpetra
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/MeshModel
      COMMAND ${AlbanyT.exe} inputEdgeDBCT.xml)
    add_test(
      NAME ${testName}_Mmodel_VtxDBC_Tpetra
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/MeshModel
      COMMAND ${AlbanyT.exe} inputVtxDBCT.xml)
    add_test(
      NAME ${testName}_Mmodel_Rename_Tpetra
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/MeshModel
      COMMAND ${AlbanyT.exe} inputRenameT.xml)
  ENDIF()
ENDIF()

