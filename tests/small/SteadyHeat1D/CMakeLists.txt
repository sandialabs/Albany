# The root name for the tests in this directory is the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if (ALBANY_EPETRA) 
  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
  # Create the test with this name and standard executable
  add_test(${testName}_Epetra ${Albany.exe} input.yaml)
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Basic;Epetra;Forward")
endif()

if (ALBANY_IFPACK2)
  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)
  # Create the test with this name and standard executable
  add_test(${testName}_Tpetra ${Albany.exe} inputT.yaml)
  set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "Basic;Tpetra;Forward")
endif ()

# Repeat process for Dakota problems if "dakota.in" exists
if (ALBANY_DAKOTA)
  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota.in)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota.in
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota.in COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota_nls.data)
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota_nls.data
                     ${CMAKE_CURRENT_BINARY_DIR}/dakota_nls.data COPYONLY)
    endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota_nls.data)
    add_test(${testName}_Dakota_Epetra ${AlbanyDakota.exe} input.yaml)
    set_tests_properties(${testName}_Dakota_Epetra PROPERTIES LABELS "Epetra;Analysis;Dakota")

    # Additional files and tests for Dakota restart testing
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dak.in
                   ${CMAKE_CURRENT_BINARY_DIR}/dak.in COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dak.restart
                   ${CMAKE_CURRENT_BINARY_DIR}/dak.restart COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_restart.yaml
                     ${CMAKE_CURRENT_BINARY_DIR}/input_restart.yaml COPYONLY)
    if (ALBANY_DAKOTA_RESTART_EXAMPLES)
      add_test(${testName}_Dakot_restarta_Epetra ${AlbanyDakota.exe} input_restart.yaml)
      set_tests_properties(${testName}_Dakota_restart_Epetra PROPERTIES LABELS "Epetra;Analysis;Dakota")
      set_tests_properties(${testName}_Dakota_restart_Epetra PROPERTIES DEPENDS  ${testName}_Dakota_Epetra)
      if(ALBANY_CTEST_TIMEOUT)
        set_tests_properties (${testName}_Dakota_restart_Epetra PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
      endif()

    endif ()
  #

  endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota.in)

endif(ALBANY_DAKOTA)

# Repeat process for Analysis problems if "inputAnalysisT.yaml" exists
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysisT.yaml)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysisT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputAnalysisT.yaml COPYONLY)

  add_test(${testName}_Analysis_Tpetra ${AlbanyAnalysis.exe} inputAnalysisT.yaml)
  set_tests_properties(${testName}_Analysis_Tpetra  PROPERTIES LABELS "Tpetra;Analysis;Basic")

  # Additional files for restart testing
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis_restartT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputAnalysis_restartT.yaml COPYONLY)

  add_test(${testName}_Analysis_restart_Tpetra ${AlbanyAnalysis.exe} inputAnalysis_restartT.yaml)
  set_tests_properties(${testName}_Analysis_restart_Tpetra  PROPERTIES LABELS "Tpetra;Analysis;Basic")
  set_tests_properties(${testName}_Analysis_restart_Tpetra  PROPERTIES DEPENDS  ${testName}_Analysis_Tpetra)
  if(ALBANY_CTEST_TIMEOUT)
    set_tests_properties (${testName}_restartT PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
  endif()

endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysisT.yaml)
