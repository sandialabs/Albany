get_filename_component (testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Tests
if (ALBANY_PANZER_EXPR_EVAL)
  # These tests require to set some input fields via a math expression.
  # If PanzerExprEval is not found in Trilinos, we can't run them.
  configure_file (${CMAKE_CURRENT_SOURCE_DIR}/input_bueler_steady.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_bueler_steady.yaml COPYONLY)
  configure_file (${CMAKE_CURRENT_SOURCE_DIR}/circle.msh
                  ${CMAKE_CURRENT_BINARY_DIR}/circle.msh COPYONLY)
  add_test (${testName}_bueler_steady   ${Albany.exe} input_bueler_steady.yaml)
  set_tests_properties(${testName}_bueler_steady PROPERTIES LABELS "LandIce;Tpetra;Forward")

  # These tests require to set some input fields via a math expression.
  # If PanzerExprEval is not found in Trilinos, we can't run them.
  configure_file (${CMAKE_CURRENT_SOURCE_DIR}/input_suite_A1.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_suite_A1.yaml COPYONLY)
  add_test (${testName}_suite_A1   ${Albany.exe} input_suite_A1.yaml)
  set_tests_properties(${testName}_suite_A1 PROPERTIES LABELS "LandIce;Tpetra;Forward")

  # These tests require to set some input fields via a math expression.
  # If PanzerExprEval is not found in Trilinos, we can't run them.
  configure_file (${CMAKE_CURRENT_SOURCE_DIR}/input_suite_A2.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_suite_A2.yaml COPYONLY)
  add_test (${testName}_suite_A2   ${Albany.exe} input_suite_A2.yaml)
  set_tests_properties(${testName}_suite_A2 PROPERTIES LABELS "LandIce;Tpetra;Forward")

  # These tests require to set some input fields via a math expression.
  # If PanzerExprEval is not found in Trilinos, we can't run them.
  configure_file (${CMAKE_CURRENT_SOURCE_DIR}/input_suite_A3.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_suite_A3.yaml COPYONLY)
  add_test (${testName}_suite_A3   ${Albany.exe} input_suite_A3.yaml)
  set_tests_properties(${testName}_suite_A3 PROPERTIES LABELS "LandIce;Tpetra;Forward")

  # These tests require to set some input fields via a math expression.
  # If PanzerExprEval is not found in Trilinos, we can't run them.
  configure_file (${CMAKE_CURRENT_SOURCE_DIR}/input_suite_A4.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_suite_A4.yaml COPYONLY)
  add_test (${testName}_suite_A4   ${Albany.exe} input_suite_A4.yaml)
  set_tests_properties(${testName}_suite_A4 PROPERTIES LABELS "LandIce;Tpetra;Forward")

  # These tests require to set some input fields via a math expression.
  # If PanzerExprEval is not found in Trilinos, we can't run them.
  configure_file (${CMAKE_CURRENT_SOURCE_DIR}/input_suite_A5.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_suite_A5.yaml COPYONLY)
  add_test (${testName}_suite_A5   ${Albany.exe} input_suite_A5.yaml)
  set_tests_properties(${testName}_suite_A5 PROPERTIES LABELS "LandIce;Tpetra;Forward")

  # These tests require to set some input fields via a math expression.
  # If PanzerExprEval is not found in Trilinos, we can't run them.
  # configure_file (${CMAKE_CURRENT_SOURCE_DIR}/input_suite_A6.yaml
  #                 ${CMAKE_CURRENT_BINARY_DIR}/input_suite_A6.yaml COPYONLY)
  # add_test (${testName}_suite_A6   ${Albany.exe} input_suite_A6.yaml)
  # set_tests_properties(${testName}_suite_A6 PROPERTIES LABELS "LandIce;Tpetra;Forward")
endif()
