# Name the test with the directory name
get_filename_component(testDirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

IF (SEACAS_EPU AND SEACAS_EXODIFF) 
  # Copy mesh files from source dir to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen.4.0 ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen.4.0 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen.4.1 ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen.4.1 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen.4.2 ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen.4.2 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen.4.3 ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen.4.3 COPYONLY)

  set (testName ${testDirName}_adj)
  IF (ALBANY_EPETRA)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal_adj.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/nodal_adj.yaml COPYONLY)

    createExoDiffTest(ATO_${testName}_Epetra nodal_adj.yaml physics_0_mitchell)
    set_tests_properties(ATO_${testName}_Epetra PROPERTIES LABELS "ATO;Epetra;Forward")
  ENDIF() 

  # Currently crashes in belos due to nans. I suspect un-initialized vector somewhere
  # IF (ALBANY_IFPACK2)
  #   configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal_adjT.yaml
  #                  ${CMAKE_CURRENT_BINARY_DIR}/nodal_adjT.yaml COPYONLY)

  #   createExoDiffTest(${testName}_Tpetra nodal_adjT.yaml physics_0_mitchellT)
  #   set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "ATO;Tpetra;Forward")
  # ENDIF() 

  set (testName ${testDirName}_fd)
  IF (ALBANY_EPETRA)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal_fd.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/nodal_fd.yaml COPYONLY)

    createExoDiffTest(ATO_${testName}_Epetra nodal_fd.yaml physics_0_mitchell)
    set_tests_properties(ATO_${testName}_Epetra PROPERTIES LABELS "ATO;Epetra;Forward")
  ENDIF() 

  # Currently crashes in belos due to nans. I suspect un-initialized vector somewhere
  # IF (ALBANY_IFPACK2)
  #   configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal_fdT.yaml
  #                  ${CMAKE_CURRENT_BINARY_DIR}/nodal_fdT.yaml COPYONLY)

  #   createExoDiffTest(${testName}_Tpetra nodal_fdT.yaml physics_0_mitchellT)
  #   set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "ATO;Tpetra;Forward")
  # ENDIF() 
ENDIF()
