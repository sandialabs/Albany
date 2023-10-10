# The root name for the tests in this directory is the directory name
get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${parentDirName}_${dirName})

# Create the test with this name and standard executable
set(testName ${testNameRoot})
if (ALBANY_EPETRA) 

  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
  add_test(${testName}_Epetra ${Albany.exe} input.yaml)
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Basic;Epetra;Forward")
endif()

if (ALBANY_IFPACK2)
  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)

  add_test(${testName}_Tpetra ${Albany.exe} inputT.yaml)
  set_tests_properties(${testName}_Tpetra  PROPERTIES LABELS "Basic;Tpetra;Forward")

  add_test(${testName}_SERIAL ${SerialAlbany.exe} inputT.yaml)
  set_tests_properties(${testName}_SERIAL PROPERTIES LABELS "Basic;Tpetra;Forward")

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT_RegressFail.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputT_RegressFail.yaml COPYONLY)
  add_test(${testName}_RegressFail ${SerialAlbany.exe} inputT_RegressFail.yaml)
  set_tests_properties(${testName}_RegressFail PROPERTIES WILL_FAIL TRUE)
  set_tests_properties(${testName}_RegressFail PROPERTIES LABELS "Basic;Tpetra;Forward;RegressFail")

  if (ALBANY_OMEGAH)
    # Copy Input file from source to binary dir
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT_Omega_h.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputT_Omega_h.yaml COPYONLY)
    # Create the tests with this name and standard executable
    add_test(${testName}_Tpetra_Omega_h ${Albany.exe} inputT_Omega_h.yaml)
    set_tests_properties(${testName}_Tpetra_Omega_h PROPERTIES LABELS "Basic;Tpetra;Forward;Omega_h")
  endif()
endif ()

if (ALBANY_MUELU_EXAMPLES)
  set (testName ${testNameRoot}_MueLu)

  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT_MueLu.yaml
             ${CMAKE_CURRENT_BINARY_DIR}/inputT_MueLu.yaml COPYONLY)
  # Create the tests with this name and standard executable
  add_test(${testName} ${Albany.exe} inputT_MueLu.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "Basic;Tpetra;Forward")

  add_test(${testName}_SERIAL ${SerialAlbany.exe} inputT_MueLu.yaml)
  set_tests_properties(${testName}_SERIAL PROPERTIES LABELS "Basic;Tpetra;Forward")

  # Regress fail test
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT_MueLu_RegressFail.yaml
             ${CMAKE_CURRENT_BINARY_DIR}/inputT_MueLu_RegressFail.yaml COPYONLY)
  add_test(${testName}_RegressFail ${SerialAlbany.exe} inputT_MueLu_RegressFail.yaml)
  set_tests_properties(${testName}_RegressFail PROPERTIES LABELS "Basic;Tpetra;Forward;RegressFail")
  set_tests_properties(${testName}_RegressFail PROPERTIES WILL_FAIL TRUE)
endif ()
