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
endif()
