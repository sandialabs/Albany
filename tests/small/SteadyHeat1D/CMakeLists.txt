# The root name for the tests in this directory is the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if (ALBANY_EPETRA) 
  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
  # Create the test with this name and standard executable
  add_test(${testName}_Epetra ${Albany.exe} input.yaml)
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Basic;Epetra;Forward")

  # Additional files for restart testing
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_restart.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_restart.yaml COPYONLY)

  add_test(${testName}_restart ${Albany.exe} input_restart.yaml)
  set_tests_properties(${testName}_restart PROPERTIES LABELS "Basic;Epetra;Forward")
  set_tests_properties(${testName}_restart PROPERTIES DEPENDS  ${testName}_Epetra)
endif()

if (ALBANY_IFPACK2)
  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)
  # Create the test with this name and standard executable
  add_test(${testName}_Tpetra ${Albany.exe} inputT.yaml)
  set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "Basic;Tpetra;Forward")
endif ()
