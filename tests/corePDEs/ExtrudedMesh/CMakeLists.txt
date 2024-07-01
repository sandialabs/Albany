if (ALBANY_IFPACK2 AND ALBANY_OMEGAH)
  # Name the test with the directory name
  get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
  get_filename_component(parentDirName ${parentPath} NAME)
  get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  set (testNameRoot ${parentDirName}_${dirName})

  # Create the test with this name and standard executable
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
  add_test(${testNameRoot} ${Albany.exe} input.yaml)
  set_tests_properties(${testNameRoot} PROPERTIES LABELS "Basic;Forward")
endif ()
