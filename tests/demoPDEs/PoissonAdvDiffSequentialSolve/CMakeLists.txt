IF (ALBANY_IFPACK2)
  # Name the test with the directory name
  get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
  get_filename_component(parentDirName ${parentPath} NAME)
  get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  set (testName ${parentDirName}_${dirName})
  
  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupled.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/coupled.yaml COPYONLY)

  # Create the test with this name and standard executable
  #add_test(${testName} ${Albany.exe} input.yaml)
  #set_tests_properties(${testName} PROPERTIES LABELS "Demo;Forward")
ENDIF()
