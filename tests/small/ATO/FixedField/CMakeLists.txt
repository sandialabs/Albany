# Create the test with this name and standard executable
IF (SEACAS_EPU AND SEACAS_EXODIFF)
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # Copy mesh files from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pcm_opt.gen ${CMAKE_CURRENT_BINARY_DIR}/pcm_opt.gen COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pcm_opt.gen.4.0 ${CMAKE_CURRENT_BINARY_DIR}/pcm_opt.gen.4.0 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pcm_opt.gen.4.1 ${CMAKE_CURRENT_BINARY_DIR}/pcm_opt.gen.4.1 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pcm_opt.gen.4.2 ${CMAKE_CURRENT_BINARY_DIR}/pcm_opt.gen.4.2 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pcm_opt.gen.4.3 ${CMAKE_CURRENT_BINARY_DIR}/pcm_opt.gen.4.3 COPYONLY)

  IF (ALBANY_EPETRA) 
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pcm_opt.yaml ${CMAKE_CURRENT_BINARY_DIR}/pcm_opt.yaml COPYONLY)
    createExoDiffTest(${testName}_Epetra pcm_opt.yaml physics_0_pcm_opt)
    set_tests_properties(${testName}_Epetra PROPERTIES LABELS "ATO;Epetra;Forward")
  ENDIF() 

  IF (ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pcm_optT.yaml ${CMAKE_CURRENT_BINARY_DIR}/pcm_optT.yaml COPYONLY)
    createExoDiffTest(${testName}_Tpetra pcm_optT.yaml physics_0_pcm_opt)
    set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "ATO;Tpetra;Forward")
  ENDIF()
ENDIF()
