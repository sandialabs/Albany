if (ALBANY_IOPX)
  # Name the test with the directory name
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # Copy mesh file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/domain.exo
                 ${CMAKE_CURRENT_BINARY_DIR}/domain.exo COPYONLY)

  if (ALBANY_EPETRA) 
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)

    add_test(${testName}_Epetra ${Albany.exe} input.yaml)
    set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Demo;Epetra;Forward")
  endif()

  if (ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT_rythmos.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputT_rythmos.yaml COPYONLY)
    add_test(${testName}_Rythmos_Tpetra ${Albany.exe} inputT_rythmos.yaml)
    set_tests_properties(${testName}_Rythmos_Tpetra PROPERTIES LABELS "Demo;Tpetra;Forward")

    if (ALBANY_TEMPUS)
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT_tempus.yaml
                     ${CMAKE_CURRENT_BINARY_DIR}/inputT_tempus.yaml COPYONLY)
      add_test(${testName}_Tempus_Tpetra ${Albany.exe} inputT_tempus.yaml)
      set_tests_properties(${testName}_Tempus_Tpetra PROPERTIES LABELS "Demo;Tpetra;Forward")
    endif()
  endif()
endif(ALBANY_IOPX)
