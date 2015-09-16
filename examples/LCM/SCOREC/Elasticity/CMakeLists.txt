#*****************************************************************//
#    Albany 2.0:  Copyright 2012 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//


# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSingleMeshT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSingleMeshT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputEdgeDBCT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputEdgeDBCT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputVtxDBCT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputVtxDBCT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputRenameT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputRenameT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputNeckingT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputNeckingT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputNeckingSingleMeshT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputNeckingSingleMeshT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSquoze.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSquoze.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSquoze2.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSquoze2.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSquozeSerial.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSquozeSerial.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSquozeSerial2.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSquozeSerial2.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/traction_assoc.txt
               ${CMAKE_CURRENT_BINARY_DIR}/traction_assoc.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vtxDBCassoc.txt
               ${CMAKE_CURRENT_BINARY_DIR}/vtxDBCassoc.txt COPYONLY)

# 2. Name the test with the directory name, prepended by SCOREC_
get_filename_component(testRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(testName "SCOREC_${testRoot}")

# 3. Create the test with this name and standard executable
IF(ENABLE_MESHMODEL)
  IF(ALBANY_IFPACK2)
    add_test(NAME ${testName}_Traction_Tpetra COMMAND ${AlbanyT.exe} inputT.xml)
    add_test(NAME ${testName}_TracT_SM COMMAND ${AlbanyT.exe} inputSingleMeshT.xml)
    add_test(NAME ${testName}_EdgeDBC_Tpetra COMMAND ${AlbanyT.exe} inputEdgeDBCT.xml)
    add_test(NAME ${testName}_VtxDBC_Tpetra COMMAND ${AlbanyT.exe} inputVtxDBCT.xml)
    add_test(NAME ${testName}_Rename_Tpetra COMMAND ${AlbanyT.exe} inputRenameT.xml)
    add_test(NAME ${testName}_Necking_Tpetra COMMAND ${AlbanyT.exe} inputNeckingT.xml)
    add_test(NAME ${testName}_NeckT_SM COMMAND ${AlbanyT.exe} inputNeckingSingleMeshT.xml)
  ENDIF()
  if (ALBANY_MUELU_EXAMPLES)
    add_test(NAME ${testName}_Pressure_Tpetra COMMAND ${AlbanyT.exe} inputSquoze2.xml)
#    add_test(NAME ${testName}_Pressure_Tpetra_SERIAL COMMAND ${SerialAlbanyT.exe} inputSquozeSerial.xml)
  ENDIF()
ENDIF()
