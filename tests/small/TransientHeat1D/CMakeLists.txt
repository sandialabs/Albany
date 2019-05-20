# The root name for the tests in this directory is the directory name
get_filename_component(testNameRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)

####################################
###          Main tests          ###
####################################

set (testName ${testNameRoot})

if (ALBANY_EPETRA) 
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)

  add_test(${testName}_Epetra ${Albany.exe} input.yaml)
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Basic;Epetra;Forward")
endif()

if (ALBANY_IFPACK2) 
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)
  add_test(${testName}_Tpetra ${Albany.exe} inputT.yaml)
  set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "Basic;Tpetra;Forward")
endif()

####################################
###        Analysis tests        ###
####################################

# Repeat process for Dakota problems
if (ALBANY_DAKOTA AND ALBANY_EPETRA)
  set (testName ${testNameRoot}_Dakota)
  if (ALBANY_IFPACK2) 
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota_list.in
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota_list.in COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputDakotaT_list.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputDakotaT_list.yaml COPYONLY)

    add_test(${testName}_Tpetra ${AlbanyDakota.exe} inputDakotaT_list.yaml)
    set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "Tpetra;Analysis;Dakota")
  endif ()
endif()

# Repeat process for Analysis problems
if (ALBANY_EPETRA)
  set (testName ${testNameRoot}_OptiPack)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputAnalysis.yaml COPYONLY)

  add_test(${testName} ${AlbanyAnalysis.exe} inputAnalysis.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "Epetra;Analysis;OptiPack")
endif()
