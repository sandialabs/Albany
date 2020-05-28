if(NOT ALBANY_PARALLEL_ONLY)
  # Name the test with the directory name
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_steady.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_steady.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_with_source.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_with_source.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/compute_errs_source_no_movie2avi.m
                 ${CMAKE_CURRENT_BINARY_DIR}/compute_errs_source_no_movie2avi.m COPYONLY)

  # Create the test with this name and standard executable
  add_test(${testName} ${Albany.exe} input.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "Demo;Tpetra;Forward")
  add_test(${testName}_with_Source ${Albany.exe} input_with_source.yaml)
  set_tests_properties(${testName}_with_Source PROPERTIES LABELS "Demo;Tpetra;Forward")
endif()
