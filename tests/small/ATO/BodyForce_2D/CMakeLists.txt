IF (DEFINED MPIMNP AND ${MPIMNP} GREATER 1 )
  SET (PARALLEL TRUE)
ELSE()
  SET (PARALLEL FALSE)
ENDIF()

IF (NOT PARALLEL AND ATO_NLOPT)

  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  IF (ALBANY_EPETRA)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.yaml ${CMAKE_CURRENT_BINARY_DIR}/mitchell.yaml COPYONLY)
    add_test(ATO_${testName}_Epetra ${Albany.exe} mitchell.yaml)
    set_tests_properties(ATO_${testName}_Epetra PROPERTIES LABELS "ATO;Epetra;Forward")
  ENDIF() 


  IF (ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchellT.yaml ${CMAKE_CURRENT_BINARY_DIR}/mitchellT.yaml COPYONLY)
    add_test(ATO_${testName}_Tpetra ${Albany.exe} mitchellT.yaml)
    set_tests_properties(ATO_${testName}_Tpetra PROPERTIES LABELS "ATO;Tpetra;Forward")
  ENDIF()

ENDIF() 
