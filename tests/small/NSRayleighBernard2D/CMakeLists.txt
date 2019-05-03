if (ALBANY_EPETRA) 
  # Name the test with the directory name
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # Copy input and mesh files from source to binary dir

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ns-m4-bKL.par
                 ${CMAKE_CURRENT_BINARY_DIR}/ns-m4-bKL.par COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ns-m4-bKL.par.4.0
                 ${CMAKE_CURRENT_BINARY_DIR}/ns-m4-bKL.par.4.0 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ns-m4-bKL.par.4.1
                 ${CMAKE_CURRENT_BINARY_DIR}/ns-m4-bKL.par.4.1 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ns-m4-bKL.par.4.2
                 ${CMAKE_CURRENT_BINARY_DIR}/ns-m4-bKL.par.4.2 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ns-m4-bKL.par.4.3
                 ${CMAKE_CURRENT_BINARY_DIR}/ns-m4-bKL.par.4.3 COPYONLY)

  # Create the test
  add_test(${testName}_Epetra ${Albany.exe} input.yaml)
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Demo;Epetra;Forward")
endif()
