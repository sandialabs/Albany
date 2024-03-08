get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Name the test with the directory name
set (testName ${parentDirName}_${dirName})

if(NOT ALBANY_PARALLEL_ONLY)
  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.g
                 ${CMAKE_CURRENT_BINARY_DIR}/cube.g COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.g.4.0
                 ${CMAKE_CURRENT_BINARY_DIR}/cube.g.4.0 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.g.4.1
                 ${CMAKE_CURRENT_BINARY_DIR}/cube.g.4.1 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.g.4.2
                 ${CMAKE_CURRENT_BINARY_DIR}/cube.g.4.2 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.g.4.3
                 ${CMAKE_CURRENT_BINARY_DIR}/cube.g.4.3 COPYONLY)

  # Create the test with this name and standard executable
  add_test(${testName} ${Albany.exe} input.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "Demo;Forward")
endif()
