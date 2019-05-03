IF (NOT ALBANY_PARALLEL_ONLY)
  # Name the test with the directory name
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_SERIAL NAME)

  ####################################
  ###            Tests             ###
  ####################################

  IF(ALBANY_EPETRA)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)

    add_test(${testName}_Epetra ${SerialAlbany.exe} input.yaml)
    set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Demo;Epetra;Forward;Serial")
  ENDIF()

  if (NOT (ALBANY_CHECK_FPE OR ALBANY_STRONG_FPE_CHECK))
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)
    # Fails with FPE if FPE checking is enabled
    add_test(${testName}_Tpetra ${SerialAlbanyT.exe} inputT.yaml)
    set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "Demo;Tpetra;Forward;Serial")
  endif ()

  if (ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTP.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputTP.yaml COPYONLY)

    add_test(${testName}_Ifpack2_Tpetra ${SerialAlbany.exe} inputTP.yaml)
    set_tests_properties(${testName}_Ifpack2_Tpetra PROPERTIES LABELS "Demo;Tpetra;Forward;Serial")
  endif()
ENDIF()
