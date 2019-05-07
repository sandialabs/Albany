if (ALBANY_EPETRA) 
  # Name the test with the directory name
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_domain_coupled.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_domain_coupled.yaml COPYONLY)

  # Create the test
  add_test(${testName}_Epetra ${Albany.exe} input_domain_coupled.yaml)
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Demo;Epetra;Forward")
endif()

