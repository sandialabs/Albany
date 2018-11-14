# 1. Copy Input file from source to binary dir
CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/input_steady.yaml
                ${CMAKE_CURRENT_BINARY_DIR}/input_steady.yaml COPYONLY)
CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/input_unsteady.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_unsteady.yaml COPYONLY)

GET_FILENAME_COMPONENT (testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#2. Tests
IF (ALBANY_PANZER_EXPR_EVAL)
  # These tests require to set some input fields via a math expression.
  # If PanzerExprEval is not found in Trilinos, we can't run them.
  ADD_TEST (${testName}_steady   ${AlbanyT.exe} input_steady.yaml)
  ADD_TEST (${testName}_unsteady ${AlbanyT.exe} input_unsteady.yaml)
ENDIF()
