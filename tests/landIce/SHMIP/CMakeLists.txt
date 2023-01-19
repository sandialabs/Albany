get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${parentDirName}_${dirName})

# These tests require to set some input fields via a math expression.
# If PanzerExprEval is not found in Trilinos, we can't run them.
if (ALBANY_PANZER_EXPR_EVAL)

  # -------------------------------------- #
  #  Ed Bueler's quasi-analytic test case  #
  # -------------------------------------- #
  configure_file (${CMAKE_CURRENT_SOURCE_DIR}/input_bueler_steady.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_bueler_steady.yaml COPYONLY)
  configure_file (${CMAKE_CURRENT_SOURCE_DIR}/circle.msh
                  ${CMAKE_CURRENT_BINARY_DIR}/circle.msh COPYONLY)
	  add_test (${testNameRoot}_bueler_steady   ${Albany.exe} input_bueler_steady.yaml)
	  set_tests_properties(${testNameRoot}_bueler_steady PROPERTIES LABELS "LandIce;Tpetra;Forward")

  # NOTE: the unsteady test produce a different solution than the steady one.
  #       It appears it takes *a lot of simulated years* for the unsteady solution
  #       to approach the steady one. And even after million of years, it seems that
  #       the water thickness is smoother and larger, compared to the SS case.
  #       This might be due to dissipation/dispersion induced by the time integrators.
  configure_file (${CMAKE_CURRENT_SOURCE_DIR}/input_bueler_unsteady.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_bueler_unsteady.yaml COPYONLY)
  add_test (${testNameRoot}_bueler_unsteady   ${Albany.exe} input_bueler_unsteady.yaml)
  set_tests_properties(${testNameRoot}_bueler_unsteady PROPERTIES LABELS "LandIce;Tpetra;Forward")

  # -------------------------------------- #
  #            SHMIP suite A               #
  # -------------------------------------- #
  if (ALBANY_LARGE_TESTS)
    configure_file (${CMAKE_CURRENT_SOURCE_DIR}/input_suite_A1.yaml
                    ${CMAKE_CURRENT_BINARY_DIR}/input_suite_A1.yaml COPYONLY)
      add_test (${testNameRoot}_suite_A1   ${Albany.exe} input_suite_A1.yaml)
    set_tests_properties(${testNameRoot}_suite_A1 PROPERTIES LABELS "LandIce;Tpetra;Forward")
  endif()

  configure_file (${CMAKE_CURRENT_SOURCE_DIR}/input_suite_A2.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_suite_A2.yaml COPYONLY)
  add_test (${testNameRoot}_suite_A2   ${Albany.exe} input_suite_A2.yaml)
  set_tests_properties(${testNameRoot}_suite_A2 PROPERTIES LABELS "LandIce;Tpetra;Forward")

  configure_file (${CMAKE_CURRENT_SOURCE_DIR}/input_suite_A3.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_suite_A3.yaml COPYONLY)
  add_test (${testNameRoot}_suite_A3   ${Albany.exe} input_suite_A3.yaml)
  set_tests_properties(${testNameRoot}_suite_A3 PROPERTIES LABELS "LandIce;Tpetra;Forward")

  configure_file (${CMAKE_CURRENT_SOURCE_DIR}/input_suite_A4.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_suite_A4.yaml COPYONLY)
  add_test (${testNameRoot}_suite_A4   ${Albany.exe} input_suite_A4.yaml)
  set_tests_properties(${testNameRoot}_suite_A4 PROPERTIES LABELS "LandIce;Tpetra;Forward")

  configure_file (${CMAKE_CURRENT_SOURCE_DIR}/input_suite_A5.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_suite_A5.yaml COPYONLY)
	  add_test (${testNameRoot}_suite_A5   ${Albany.exe} input_suite_A5.yaml)
  set_tests_properties(${testNameRoot}_suite_A5 PROPERTIES LABELS "LandIce;Tpetra;Forward")

  # -------------------------------------- #
  #     Ian Hewitt's rectancular ramp      #
  # -------------------------------------- #
  configure_file (${CMAKE_CURRENT_SOURCE_DIR}/input_hewitt.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_hewitt.yaml COPYONLY)
  add_test (${testNameRoot}_hewitt   ${Albany.exe} input_hewitt.yaml)
  set_tests_properties(${testNameRoot}_hewitt PROPERTIES LABELS "LandIce;Tpetra;Forward")
endif()
