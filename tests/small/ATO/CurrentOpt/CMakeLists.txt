IF (DEFINED MPIMNP AND ${MPIMNP} GREATER 1 )
  SET (PARALLEL TRUE)
ELSE()
  SET (PARALLEL FALSE)
ENDIF()

IF (NOT PARALLEL AND ATO_NLOPT)

  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # Copy mesh files from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2d_ebw.gen     ${CMAKE_CURRENT_BINARY_DIR}/2d_ebw.gen COPYONLY)

  IF (ALBANY_EPETRA)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2d_ebw.yaml ${CMAKE_CURRENT_BINARY_DIR}/2d_ebw.yaml COPYONLY)
    add_test(ATO_${testName}_Epetra ${Albany.exe} 2d_ebw.yaml)
    set_tests_properties(ATO_${testName}_Epetra PROPERTIES LABELS "ATO;Epetra;Forward")
  ENDIF() 


  IF (ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2d_ebwT.yaml ${CMAKE_CURRENT_BINARY_DIR}/2d_ebwT.yaml COPYONLY)
    add_test(ATO_${testName}_Tpetra ${Albany.exe} 2d_ebwT.yaml)
    set_tests_properties(ATO_${testName}_Tpetra PROPERTIES LABELS "ATO;Tpetra;Forward")
  ENDIF()

ENDIF() 
