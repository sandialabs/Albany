if (ALBANY_IOPX)
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_16km NAME)

  if (ALBANY_EPETRA) 
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputML3.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputML3.yaml COPYONLY)

    add_test(${testName}_ML ${Albany8.exe} inputML3.yaml)
    set_tests_properties(${testName}_ML PROPERTIES LABELS "LandIce;Epetra;Forward")
  endif()

  if(ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMueLuShort3.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputMueLuShort3.yaml COPYONLY)

    add_test(${testName}_MueLu ${Albany8.exe} inputMueLuShort3.yaml)
    set_tests_properties(${testName}_MueLu PROPERTIES LABELS "LandIce;Tpetra;Forward")
  endif()
endif()
