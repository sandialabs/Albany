get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if (ALBANY_EPETRA AND ALBANY_IOPX)
  # Copy input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_enthalpy_test.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_enthalpy_test.yaml COPYONLY)

  # Create the test
  add_test(${testName}_GIS ${Albany.exe} input_enthalpy_test.yaml)
  set_tests_properties(${testName}_GIS PROPERTIES LABELS "LandIce;Epetra;Forward")
  
  
  # Copy input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_kleiner_B.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_kleiner_B.yaml COPYONLY)

  # Create the test
  add_test(${testName}_Kleiner_B ${SerialAlbany.exe} input_kleiner_B.yaml)
  set_tests_properties(${testName}_Kleiner_B PROPERTIES LABELS "LandIce;Epetra;Forward,Serial")
  
  
  # Copy input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_kleiner_A.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_kleiner_A.yaml COPYONLY)

  # Create the test
  add_test(${testName}_Kleiner_A ${SerialAlbany.exe} input_kleiner_A.yaml)
  set_tests_properties(${testName}_Kleiner_A PROPERTIES LABELS "LandIce;Epetra;Forward,Serial")
endif()

