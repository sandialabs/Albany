if (ALBANY_EPETRA) 
  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputA.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputA.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputB.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputB.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputC.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputC.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputD.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputD.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/beam.pg
                 ${CMAKE_CURRENT_BINARY_DIR}/beam.pg COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)

  # 2. Name the test with the directory name
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  add_test(${testName}_A ${Albany.exe} inputA.yaml)
  set_tests_properties(${testName}_A PROPERTIES LABELS "Basic;Epetra;Forward")

  add_test(${testName}_B ${Albany.exe} inputB.yaml)
  set_tests_properties(${testName}_B PROPERTIES LABELS "Basic;Epetra;Forward")

  add_test(${testName}_C ${Albany.exe} inputC.yaml)
  set_tests_properties(${testName}_C PROPERTIES LABELS "Basic;Epetra;Forward")

  add_test(${testName}_D ${Albany.exe} inputD.yaml)
  set_tests_properties(${testName}_D PROPERTIES LABELS "Basic;Epetra;Forward")
endif()
