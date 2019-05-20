# The root name for the tests in this directory is the directory name
get_filename_component(testNameRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)

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

# Repeat process for Dakota problems
if (ALBANY_DAKOTA)
  set (testName ${testNameRoot}_Dakota)

  if(ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota.in
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota.in COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota_nls.data
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota_nls.data COPYONLY)

    add_test(${testName} ${AlbanyDakota.exe} inputT.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "Tpetra;Analysis;Dakota")
  endif()

  if(ALBANY_EPETRA)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota_nls.data
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota_nls.data COPYONLY)

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputDakota_list.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputDakota_list.yaml COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota_list.in
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota_list.in COPYONLY)
    add_test(${testName}_list ${AlbanyDakota.exe} inputDakota_list.yaml)
    set_tests_properties(${testName}_list PROPERTIES LABELS "Epetra;Analysis;Dakota")

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputDakota_uniform.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputDakota_uniform.yaml COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota_uniform.in
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota_uniform.in COPYONLY)
    add_test(${testName}_uniform ${AlbanyDakota.exe} inputDakota_uniform.yaml)
    set_tests_properties(${testName}_uniform PROPERTIES LABELS "Epetra;Analysis;Dakota")
  endif()
endif()
