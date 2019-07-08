get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
if (ALBANY_EPETRA) 

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_dome.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_dome.yaml COPYONLY)

  add_test(${testName}_Dome ${Albany.exe} input_dome.yaml)
  set_tests_properties(${testName}_Dome PROPERTIES LABELS "LandIce;Epetra;Forward")
endif()

