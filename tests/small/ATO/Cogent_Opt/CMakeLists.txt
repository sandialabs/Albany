IF (SEACAS_EPU AND SEACAS_EXODIFF)
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # Copy mesh files from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen.4.0 ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen.4.0 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen.4.1 ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen.4.1 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen.4.2 ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen.4.2 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen.4.3 ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen.4.3 COPYONLY)

  IF (ALBANY_EPETRA) 
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.yaml ${CMAKE_CURRENT_BINARY_DIR}/mitchell.yaml COPYONLY)
    createExoDiffAlgebraTest(ATO_${testName}_Epetra mitchell.yaml physics_0_mitchell)
    set_tests_properties(ATO_${testName}_Epetra PROPERTIES LABELS "ATO;Epetra;Forward")
  ENDIF() 

  IF (ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchellT.yaml ${CMAKE_CURRENT_BINARY_DIR}/mitchellT.yaml COPYONLY)
    createExoDiffAlgebraTest(ATO_${testName}_Tpetra mitchellT.yaml physics_0_mitchellT)
    set_tests_properties(ATO_${testName}_Tpetra PROPERTIES LABELS "ATO;Tpetra;Forward")
  ENDIF()
ENDIF()
