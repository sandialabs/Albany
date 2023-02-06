# Name the test with the directory name
get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${parentDirName}_${dirName})

# Copy mesh files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vortex.exo
               ${CMAKE_CURRENT_BINARY_DIR}/vortex.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vortex.exo.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/vortex.exo.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vortex.exo.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/vortex.exo.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vortex.exo.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/vortex.exo.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vortex.exo.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/vortex.exo.4.3 COPYONLY)

####################################
###         Steady tests         ###
####################################

if (ALBANY_EPETRA)
  set (testName ${testNameRoot}_Steady)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)

  add_test(${testName} ${Albany.exe} input.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "Demo;Epetra;Forward")

endif()

####################################
###       Transient tests        ###
####################################

set (testName ${testNameRoot}_TransientIRK)

if (ALBANY_EPETRA)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTransientRK.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputTransientRK.yaml COPYONLY)

  add_test(${testName}_Epetra ${Albany.exe} inputTransientRK.yaml)
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Demo;Epetra;Forward")
endif(ALBANY_EPETRA)

if (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTransientRKT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputTransientRKT.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTransientRKT_RegressFail.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputTransientRKT_RegressFail.yaml COPYONLY)

  add_test(${testName}_Tpetra ${Albany.exe} inputTransientRKT.yaml)
  set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "Demo;Tpetra;Forward")

  add_test(NAME ${testName}_Tpetra_RegressFail
           COMMAND ${Albany.exe} inputTransientRKT_RegressFail.yaml)
  set_tests_properties(${testName}_Tpetra_RegressFail PROPERTIES WILL_FAIL TRUE)
  set_tests_properties(${testName}_Tpetra_RegressFail PROPERTIES LABELS "Demo;Tpetra;Forward;RegressFail")
endif(ALBANY_IFPACK2)
