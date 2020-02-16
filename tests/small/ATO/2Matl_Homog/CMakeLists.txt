# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
IF (SEACAS_EPU AND SEACAS_EXODIFF)

  # Copy mesh files from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/padded.gen     ${CMAKE_CURRENT_BINARY_DIR}/padded.gen COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/padded.gen.4.0 ${CMAKE_CURRENT_BINARY_DIR}/padded.gen.4.0 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/padded.gen.4.1 ${CMAKE_CURRENT_BINARY_DIR}/padded.gen.4.1 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/padded.gen.4.2 ${CMAKE_CURRENT_BINARY_DIR}/padded.gen.4.2 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/padded.gen.4.3 ${CMAKE_CURRENT_BINARY_DIR}/padded.gen.4.3 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.gen     ${CMAKE_CURRENT_BINARY_DIR}/cube.gen COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.gen.4.0 ${CMAKE_CURRENT_BINARY_DIR}/cube.gen.4.0 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.gen.4.1 ${CMAKE_CURRENT_BINARY_DIR}/cube.gen.4.1 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.gen.4.2 ${CMAKE_CURRENT_BINARY_DIR}/cube.gen.4.2 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.gen.4.3 ${CMAKE_CURRENT_BINARY_DIR}/cube.gen.4.3 COPYONLY)

  IF (ALBANY_EPETRA) 
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3D.yaml ${CMAKE_CURRENT_BINARY_DIR}/mitchell3D.yaml COPYONLY)
    createExoDiffTest(ATO_${testName}_Epetra mitchell3D.yaml physics_0_padded)
    set_tests_properties(ATO_${testName}_Epetra PROPERTIES LABELS "ATO;Epetra;Forward")
  ENDIF() 

  IF (ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3DT.yaml ${CMAKE_CURRENT_BINARY_DIR}/mitchell3DT.yaml COPYONLY)
    createExoDiffTest(ATO_${testName}_Tpetra mitchell3DT.yaml physics_0_paddedT)
    set_tests_properties(ATO_${testName}_Tpetra PROPERTIES LABELS "ATO;Tpetra;Forward")
  ENDIF()
ENDIF()
