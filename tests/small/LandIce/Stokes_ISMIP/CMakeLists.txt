get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}  NAME)
if (ALBANY_EPETRA) 

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_ismip-hom_testA.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_ismip-hom_testA.yaml COPYONLY)

  add_test(${testName}_TestA ${Albany.exe} input_ismip-hom_testA.yaml)
  set_tests_properties(${testName}_TestA PROPERTIES LABELS "LandIce;Epetra;Forward")
endif()
