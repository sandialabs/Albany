IF (DEFINED MPIMNP AND ${MPIMNP} GREATER 1 )
  SET (PARALLEL TRUE)
ELSE()
  SET (PARALLEL FALSE)
ENDIF()

IF (NOT PARALLEL AND ATO_NLOPT)

  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # Copy mesh files from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RVE_restart.gen     ${CMAKE_CURRENT_BINARY_DIR}/RVE_restart.gen COPYONLY)

  IF (ALBANY_EPETRA)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal.yaml ${CMAKE_CURRENT_BINARY_DIR}/nodal.yaml COPYONLY)
    add_test(${testName}_Epetra ${Albany.exe} nodal.yaml)
    set_tests_properties(${testName}_Epetra PROPERTIES LABELS "ATO;Epetra;Forward")
  ENDIF() 
ENDIF() 
