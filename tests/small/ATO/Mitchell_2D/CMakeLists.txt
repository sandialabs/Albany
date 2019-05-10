# Create the test with this name and standard executable
IF (SEACAS_EPU AND SEACAS_EXODIFF)
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # Copy mesh files from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen.4.0 ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen.4.0 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen.4.1 ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen.4.1 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen.4.2 ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen.4.2 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen.4.3 ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen.4.3 COPYONLY)

  IF (ALBANY_EPETRA) 
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal.yaml ${CMAKE_CURRENT_BINARY_DIR}/nodal.yaml COPYONLY)
    createExoDiffTest(${testName}_Epetra nodal.yaml physics_0_mitchell)
    set_tests_properties(${testName}_Epetra PROPERTIES LABELS "ATO;Epetra;Forward")
  ENDIF() 

  IF (ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodalT.yaml ${CMAKE_CURRENT_BINARY_DIR}/nodalT.yaml COPYONLY)
    createExoDiffTest(${testName}_Tpetra nodalT.yaml physics_0_mitchell)
    set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "ATO;Tpetra;Forward")
  ENDIF()
ENDIF()
