#*****************************************************************//
#    Albany 3.0:  Copyright 2016 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//


# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSingleMeshT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSingleMeshT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputEdgeDBCT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputEdgeDBCT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputVtxDBCT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputVtxDBCT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputRenameT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputRenameT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputNeckingT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputNeckingT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputNeckingSingleMeshT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputNeckingSingleMeshT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSquoze.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSquoze.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSquoze2.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSquoze2.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSquozeSerial.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSquozeSerial.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSquozeSerial2.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSquozeSerial2.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/traction_assoc.txt
               ${CMAKE_CURRENT_BINARY_DIR}/traction_assoc.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vtxDBCassoc.txt
               ${CMAKE_CURRENT_BINARY_DIR}/vtxDBCassoc.txt COPYONLY)

# 2. Name the test with the directory name, prepended by SCOREC_
get_filename_component(testRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(testName "SCOREC_${testRoot}")

# 3. Create the test with this name and standard executable
IF(ALBANY_IFPACK2)
  add_test(NAME ${testName}_Traction_Tpetra COMMAND ${Albany.exe} inputT.yaml)
  set_tests_properties(${testName}_Traction_Tpetra PROPERTIES LABELS "LCM;Tpetra;Forward;SCOREC")
  add_test(NAME ${testName}_TracT_SM COMMAND ${Albany.exe} inputSingleMeshT.yaml)
  set_tests_properties(${testName}_TracT_SM PROPERTIES LABELS "LCM;Tpetra;Forward;SCOREC")
  add_test(NAME ${testName}_EdgeDBC_Tpetra COMMAND ${Albany.exe} inputEdgeDBCT.yaml)
  set_tests_properties(${testName}_EdgeDBC_Tpetra PROPERTIES LABELS "LCM;Tpetra;Forward;SCOREC")
  add_test(NAME ${testName}_VtxDBC_Tpetra COMMAND ${Albany.exe} inputVtxDBCT.yaml)
  set_tests_properties(${testName}_VtxDBC_Tpetra PROPERTIES LABELS "LCM;Tpetra;Forward;SCOREC")
  add_test(NAME ${testName}_Rename_Tpetra COMMAND ${Albany.exe} inputRenameT.yaml)
  set_tests_properties(${testName}_Rename_Tpetra PROPERTIES LABELS "LCM;Tpetra;Forward;SCOREC")
  add_test(NAME ${testName}_Necking_Tpetra COMMAND ${Albany.exe} inputNeckingT.yaml)
  set_tests_properties(${testName}_Necking_Tpetra PROPERTIES LABELS "LCM;Tpetra;Forward;SCOREC")
  add_test(NAME ${testName}_NeckT_SM COMMAND ${Albany.exe} inputNeckingSingleMeshT.yaml)
  set_tests_properties(${testName}_NeckT_SM PROPERTIES LABELS "LCM;Tpetra;Forward;SCOREC")
ENDIF()
if (ALBANY_MUELU_EXAMPLES)
#  add_test(NAME ${testName}_Pressure_Tpetra COMMAND ${Albany.exe} inputSquoze2.yaml)
#  add_test(NAME ${testName}_Pressure_Tpetra_SERIAL COMMAND ${SerialAlbany.exe} inputSquozeSerial.yaml)
ENDIF()
