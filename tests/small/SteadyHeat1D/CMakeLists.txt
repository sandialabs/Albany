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

# Repeat process for Analysis problems
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysisT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputAnalysisT.yaml COPYONLY)

add_test(${testName}_Analysis ${AlbanyAnalysis.exe} inputAnalysisT.yaml)
set_tests_properties(${testName}_Analysis PROPERTIES LABELS "Tpetra;Analysis;Basic")

# Additional files for restart testing
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis_restartT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputAnalysis_restartT.yaml COPYONLY)

add_test(${testName}_Analysis_restart ${AlbanyAnalysis.exe} inputAnalysis_restartT.yaml)
set_tests_properties(${testName}_Analysis_restart PROPERTIES LABELS "Tpetra;Analysis;Basic")
set_tests_properties(${testName}_Analysis_restart PROPERTIES DEPENDS  ${testName}_Analysis_Tpetra)
if(ALBANY_CTEST_TIMEOUT)
  set_tests_properties (${testName}_Analysis_restart PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
endif()
