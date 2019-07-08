get_filename_component(testNameRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if (ALBANY_EPETRA AND ALBANY_IOPX)

  # Copy input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_FO_Thermo_test.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_FO_Thermo_test.yaml COPYONLY)
                 
                   # Copy input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_FO_Thermo_wet_bed_test.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_FO_Thermo_wet_bed_test.yaml COPYONLY)

  # Create the tests
  set (testName ${testNameRoot})
  add_test(${testName} ${Albany.exe} input_FO_Thermo_test.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Epetra;Forward")
  
  set (testName ${testNameRoot}_Wet_Bed)
  add_test(${testName} ${Albany.exe} input_FO_Thermo_wet_bed_test.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Epetra;Forward")
endif()
