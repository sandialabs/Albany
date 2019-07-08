if (ALBANY_EPETRA) 
  # Name the test with the directory name
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_2DTaylorGreenVortex_SDBC  NAME)

  # Create the test with this name and standard executable
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2D_taylorGreenVortex.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input2D_taylorGreenVortex.yaml COPYONLY)

  add_test(${testName} ${Albany.exe} input2D_taylorGreenVortex.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "Demo;Epetra;Forward")
endif()

