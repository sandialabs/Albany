# Name the test with the directory name
get_filename_component(testNameRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)

####################################
###          Poly test           ###
####################################

set(testName ${testNameRoot}_Poly)

if (ALBANY_EPETRA) 
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
  add_test(${testName}_Epetra ${Albany.exe} input.yaml)
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "LandIce;Epetra;Forward")
endif()

if (ALBANY_IFPACK2) 
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)
  add_test(${testName}_Tpetra ${Albany.exe} inputT.yaml)
  set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "LandIce;Tpetra;Forward")
endif()

####################################
###     Couette Periodic test    ###
####################################

set(testName ${testNameRoot}_CouettePeriodicX)

if (ALBANY_EPETRA) 
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_couette_periodic.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_couette_periodic.yaml COPYONLY)
  add_test(${testName} ${Albany.exe} input_couette_periodic.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Epetra;Forward")
endif()

####################################
###    SinSin PeriodicXY test    ###
####################################

set(testName ${testNameRoot}_SinSinPeriodicXY)

if (ALBANY_EPETRA) 
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sinsin.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_sinsin.yaml COPYONLY)
  add_test(${testName} ${Albany.exe} input_sinsin.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Epetra;Forward")
endif()

####################################
###     SinSin Dirichlet test    ###
####################################

set(testName ${testNameRoot}_SinSinDirichlet)

if (ALBANY_EPETRA) 
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sinsin_dirichlet.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_sinsin_dirichlet.yaml COPYONLY)
  add_test(${testName} ${Albany.exe} input_sinsin_dirichlet.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Epetra;Forward")
endif()

####################################
###    SinCosZ Dirichlet test    ###
####################################

set(testName ${testNameRoot}_SinCosZ3DDirichlet)

if (ALBANY_EPETRA) 
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sincosz_dirichlet.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_sincosz_dirichlet.yaml COPYONLY)
  add_test(${testName} ${Albany.exe} input_sincosz_dirichlet.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Epetra;Forward")
endif()

####################################
###    SinCosZ Periodic test     ###
####################################

set(testName ${testNameRoot}_SinCosZ3DPeriodicXY)

if (ALBANY_EPETRA) 
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sincosz.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_sincosz.yaml COPYONLY)
  add_test(${testName} ${Albany.exe} input_sincosz.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Epetra;Forward")
endif()

####################################
###       SinCos Glen test       ###
####################################

set(testName ${testNameRoot}_SinCosGlen)

if (ALBANY_EPETRA) 
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sincos_glen.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_sincos_glen.yaml COPYONLY)
  add_test(${testName} ${Albany.exe} input_sincos_glen.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Epetra;Forward")
endif()

####################################
###      SinExp Basal test       ###
####################################

set(testName ${testNameRoot}_SinExpBasal)

if (ALBANY_EPETRA) 
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sinexp_basal.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_sinexp_basal.yaml COPYONLY)
  add_test(${testName} ${Albany.exe} input_sinexp_basal.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Epetra;Forward")
endif()
