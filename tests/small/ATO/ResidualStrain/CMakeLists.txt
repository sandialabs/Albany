IF (DEFINED MPIMNP AND ${MPIMNP} GREATER 1 )
  SET (PARALLEL TRUE)
ELSE()
  SET (PARALLEL FALSE)
ENDIF()

IF (NOT PARALLEL AND ATO_NLOPT)
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # Copy mesh files from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen     ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen COPYONLY)

  IF (ALBANY_EPETRA)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.yaml ${CMAKE_CURRENT_BINARY_DIR}/mitchell.yaml COPYONLY)
    add_test(${testName}_Epetra ${Albany.exe} mitchell.yaml)
    set_tests_properties(${testName}_Epetra PROPERTIES LABELS "ATO;Epetra;Forward")
  ENDIF() 


  IF (ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchellT.yaml ${CMAKE_CURRENT_BINARY_DIR}/mitchellT.yaml COPYONLY)
    add_test(${testName}_Tpetra ${Albany.exe} mitchellT.yaml)
    set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "ATO;Tpetra;Forward")
  ENDIF()
ENDIF() 
