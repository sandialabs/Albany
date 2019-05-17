if (ALBANY_EPETRA) 
  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTrans.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputTrans.yaml COPYONLY)
  # Copy materials file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/qdot.dat
                 ${CMAKE_CURRENT_BINARY_DIR}/qdot.dat COPYONLY)

  # Copy mesh files from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ebs.exo
                 ${CMAKE_CURRENT_BINARY_DIR}/ebs.exo COPYONLY)

  # Name the test with the directory name
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # Create the tests

  add_test(${testName}_Steady ${SerialAlbany.exe} input.yaml)
  set_tests_properties(${testName}_Steady PROPERTIES LABELS "Basic;Epetra;Forward")

  add_test(${testName}_Transient ${SerialAlbany.exe} inputTrans.yaml)
  set_tests_properties(${testName}_Transient PROPERTIES DEPENDS ${testName}_Steady)
  set_tests_properties(${testName}_Transient PROPERTIES LABELS "Basic;Epetra;Forward")
endif()
