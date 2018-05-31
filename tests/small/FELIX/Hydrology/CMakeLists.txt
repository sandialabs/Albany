# 1. Copy Input file from source to binary dir
CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/input_steady.xml
                ${CMAKE_CURRENT_BINARY_DIR}/input_steady.xml COPYONLY)
CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/input_unsteady.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_unsteady.xml COPYONLY)

GET_FILENAME_COMPONENT (testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#2. Tests
IF (ALBANY_PANZER_EXPR_EVAL)
  # These tests require to set some input fields via a math expression.
  # If PanzerExprEval is not found in Trilinos, we can't run them.
  ADD_TEST (${testName}_steady   ${AlbanyT.exe} input_steady.xml)
  ADD_TEST (${testName}_unsteady ${AlbanyT.exe} input_unsteady.xml)
ENDIF()
