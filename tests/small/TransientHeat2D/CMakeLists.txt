# The root name for the tests in this directory is the directory name
get_filename_component(testNameRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)


####################################
###         Tempus tests         ###
####################################


set (testName ${testNameRoot}_Tempus)

if (ALBANY_EPETRA) 
  set (testName ${testNameRoot})

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)

  add_test(${testName} ${Albany.exe} input.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "Basic;Epetra;Forward")
endif()

if (ALBANY_TEMPUS AND ALBANY_IFPACK2)
  # BE test
  set (testName ${testNameRoot}_Tempus_BackwardEuler_NOXSolver)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tempus_be_nox_solver.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/tempus_be_nox_solver.yaml COPYONLY)

  add_test(${testName}_Tpetra ${Albany.exe} tempus_be_nox_solver.yaml)
  set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "Basic;Tempus;Tpetra;Forward")

  # RK 4 test
  set (testName ${testNameRoot}_Tempus_RK4)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tempus_rk4.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/tempus_rk4.yaml COPYONLY)
  add_test(${testName} ${Albany.exe} tempus_rk4.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "Basic;Tempus;Tpetra;Forward")
endif ()
