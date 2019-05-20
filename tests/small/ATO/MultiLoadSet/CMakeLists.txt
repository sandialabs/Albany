IF (ATO_NLOPT)
  get_filename_component(testNameRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # Copy mesh files from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen     ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen.4.0 ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen.4.0 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen.4.1 ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen.4.1 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen.4.2 ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen.4.2 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen.4.3 ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen.4.3 COPYONLY)

  set (testName ${testNameRoot}_oc)
  IF (ALBANY_EPETRA)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal_oc.yaml ${CMAKE_CURRENT_BINARY_DIR}/nodal_oc.yaml COPYONLY)
    add_test(${testName}_Epetra ${Albany.exe} nodal_oc.yaml)
    set_tests_properties(${testName}_Epetra PROPERTIES LABELS "ATO;Epetra;Forward")
  ENDIF() 


  IF (ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal_ocT.yaml ${CMAKE_CURRENT_BINARY_DIR}/nodal_ocT.yaml COPYONLY)
    add_test(${testName}_Tpetra ${Albany.exe} nodal_ocT.yaml)
    set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "ATO;Tpetra;Forward")
  ENDIF()

  IF (ATO_NLOPT)
    set (testName ${testNameRoot}_nlopt)
    IF (ALBANY_EPETRA)
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal_nlopt.yaml ${CMAKE_CURRENT_BINARY_DIR}/nodal_nlopt.yaml COPYONLY)
      add_test(${testName}_Epetra ${Albany.exe} nodal_nlopt.yaml)
      set_tests_properties(${testName}_Epetra PROPERTIES LABELS "ATO;Epetra;Forward")
    ENDIF() 

    IF (ALBANY_IFPACK2)
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal_nloptT.yaml ${CMAKE_CURRENT_BINARY_DIR}/nodal_nloptT.yaml COPYONLY)
      add_test(${testName}_Tpetra ${Albany.exe} nodal_nloptT.yaml)
      set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "ATO;Tpetra;Forward")
    ENDIF()
  ENDIF()

ENDIF() 
