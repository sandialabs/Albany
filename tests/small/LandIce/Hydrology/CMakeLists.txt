get_filename_component (testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Tests
if (ALBANY_PANZER_EXPR_EVAL)
  # These tests require to set some input fields via a math expression.
  # If PanzerExprEval is not found in Trilinos, we can't run them.
  configure_file (${CMAKE_CURRENT_SOURCE_DIR}/input_steady.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_steady.yaml COPYONLY)
  add_test (${testName}_steady   ${Albany.exe} input_steady.yaml)
  set_tests_properties(${testName}_steady PROPERTIES LABELS "LandIce;Tpetra;Forward")

  configure_file (${CMAKE_CURRENT_SOURCE_DIR}/input_unsteady.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_unsteady.yaml COPYONLY)
  add_test (${testName}_unsteady ${Albany.exe} input_unsteady.yaml)
  set_tests_properties(${testName}_steady PROPERTIES LABELS "LandIce;Tpetra;Forward")
endif()
