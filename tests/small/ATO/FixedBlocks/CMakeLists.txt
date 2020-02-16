# Create the test with this name and standard executable
IF (SEACAS_EPU AND SEACAS_EXODIFF)
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # Copy mesh files from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fblock.gen ${CMAKE_CURRENT_BINARY_DIR}/fblock.gen COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fblock.gen.4.0 ${CMAKE_CURRENT_BINARY_DIR}/fblock.gen.4.0 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fblock.gen.4.1 ${CMAKE_CURRENT_BINARY_DIR}/fblock.gen.4.1 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fblock.gen.4.2 ${CMAKE_CURRENT_BINARY_DIR}/fblock.gen.4.2 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fblock.gen.4.3 ${CMAKE_CURRENT_BINARY_DIR}/fblock.gen.4.3 COPYONLY)

  IF (ALBANY_EPETRA) 
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fblock.yaml ${CMAKE_CURRENT_BINARY_DIR}/fblock.yaml COPYONLY)
    createExoDiffTest(ATO_${testName}_Epetra fblock.yaml physics_0_fblock)
    set_tests_properties(ATO_${testName}_Epetra PROPERTIES LABELS "ATO;Epetra;Forward")
  ENDIF() 

  IF (ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fblockT.yaml ${CMAKE_CURRENT_BINARY_DIR}/fblockT.yaml COPYONLY)
    createExoDiffTest(ATO_${testName}_Tpetra fblockT.yaml physics_0_fblockT)
    set_tests_properties(ATO_${testName}_Tpetra PROPERTIES LABELS "ATO;Tpetra;Forward")
  ENDIF()
ENDIF()
