get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Copy mesh files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fblock_restart.exo     ${CMAKE_CURRENT_BINARY_DIR}/fblock_restart.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fblock_restart.exo.4.0 ${CMAKE_CURRENT_BINARY_DIR}/fblock_restart.exo.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fblock_restart.exo.4.1 ${CMAKE_CURRENT_BINARY_DIR}/fblock_restart.exo.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fblock_restart.exo.4.2 ${CMAKE_CURRENT_BINARY_DIR}/fblock_restart.exo.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fblock_restart.exo.4.3 ${CMAKE_CURRENT_BINARY_DIR}/fblock_restart.exo.4.3 COPYONLY)

IF (ALBANY_EPETRA)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fblock_restart.yaml ${CMAKE_CURRENT_BINARY_DIR}/fblock_restart.yaml COPYONLY)
  add_test(${testName}_Epetra ${Albany.exe} fblock_restart.yaml)
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "ATO;Epetra;Forward")
ENDIF() 

IF (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fblock_restartT.yaml ${CMAKE_CURRENT_BINARY_DIR}/fblock_restartT.yaml COPYONLY)
  add_test(${testName}_Tpetra ${Albany.exe} fblock_restartT.yaml)
  set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "ATO;Tpetra;Forward")
ENDIF()
