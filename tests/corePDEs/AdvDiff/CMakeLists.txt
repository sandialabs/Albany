IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_IFPACK2)
  # Name the test with the directory name
  get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
  get_filename_component(parentDirName ${parentPath} NAME)
  get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  set (testName ${parentDirName}_${dirName})

  # Create the test with this name and standard executable
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
  add_test(${testName} ${SerialAlbany.exe} input.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "Demo;Forward;Serial")

  # Create adaptivity test
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_adapt.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_adapt.yaml COPYONLY)
  add_test(${testName}_adapt ${SerialAlbany.exe} input_adapt.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "Demo;Forward;Serial;Adapt")
ENDIF ()
