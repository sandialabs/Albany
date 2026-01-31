# The root name for the tests in this directory is the directory name
get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${parentDirName}_${dirName})

# Create the test with this name and standard executable
set(testName ${testNameRoot})

if (ALBANY_IFPACK2)
  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)

  add_test(${testName} ${Albany.exe} input.yaml)
  set_tests_properties(${testName}  PROPERTIES LABELS "Basic;Forward")

  add_test(${testName}_SERIAL ${SerialAlbany.exe} input.yaml)
  set_tests_properties(${testName}_SERIAL PROPERTIES LABELS "Basic;Forward")

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_RegressFail.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_RegressFail.yaml COPYONLY)
  add_test(${testName}_RegressFail ${SerialAlbany.exe} input_RegressFail.yaml)
  set_tests_properties(${testName}_RegressFail PROPERTIES WILL_FAIL TRUE)
  set_tests_properties(${testName}_RegressFail PROPERTIES LABELS "Basic;Forward;RegressFail")

  if (ALBANY_OMEGAH)
    # Copy Input file from source to binary dir
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_Omega_h.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_Omega_h.yaml COPYONLY)
    # Create the tests with this name and standard executable
    add_test(${testName}_Omega_h ${Albany.exe} input_Omega_h.yaml)
    set_tests_properties(${testName}_Omega_h PROPERTIES LABELS "Basic;Forward;Omega_h")
    add_test(${testName}_Omega_h_Serial ${SerialAlbany.exe} input_Omega_h.yaml)
    set_tests_properties(${testName}_Omega_h_Serial PROPERTIES LABELS "Basic;Forward;Omega_h")
  endif()
endif ()

if (ALBANY_MUELU_EXAMPLES)
  set (testName ${testNameRoot}_MueLu)

  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_MueLu.yaml
             ${CMAKE_CURRENT_BINARY_DIR}/input_MueLu.yaml COPYONLY)
  # Create the tests with this name and standard executable
  add_test(${testName} ${Albany.exe} input_MueLu.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "Basic;Forward")

  add_test(${testName}_SERIAL ${SerialAlbany.exe} input_MueLu.yaml)
  set_tests_properties(${testName}_SERIAL PROPERTIES LABELS "Basic;Forward")

  # Regress fail test
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_MueLu_RegressFail.yaml
             ${CMAKE_CURRENT_BINARY_DIR}/input_MueLu_RegressFail.yaml COPYONLY)
  add_test(${testName}_RegressFail ${SerialAlbany.exe} input_MueLu_RegressFail.yaml)
  set_tests_properties(${testName}_RegressFail PROPERTIES LABELS "Basic;Forward;RegressFail")
  set_tests_properties(${testName}_RegressFail PROPERTIES WILL_FAIL TRUE)
endif ()
