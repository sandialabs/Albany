get_filename_component(testNameRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if (ALBANY_EPETRA AND ALBANY_IOPX)

  # Copy input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_FO_Thermo_dry_bed_test.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_FO_Thermo_dry_bed_test.yaml COPYONLY)
                 
  # Copy input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_FO_Thermo_wet_bed_test.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_FO_Thermo_wet_bed_test.yaml COPYONLY)

  # Create the tests
  set (testName ${testNameRoot}_Dry_Bed)
  add_test(${testName} ${Albany.exe} input_FO_Thermo_dry_bed_test.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Epetra;Forward")
  
  set (testName ${testNameRoot}_Wet_Bed)
  add_test(${testName} ${Albany.exe} input_FO_Thermo_wet_bed_test.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Epetra;Forward")
endif()

if (ALBANY_FROSCH)
  # Copy input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_FO_Thermo_wet_bed_test_FROSch.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_FO_Thermo_wet_bed_test_FROSch.yaml COPYONLY)

  # Create the tests
  set (testName ${testNameRoot}_Wet_Bed_FROSch)
  add_test(${testName} ${Albany.exe} input_FO_Thermo_wet_bed_test_FROSch.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Tpetra;Forward;FROSch")
endif()
