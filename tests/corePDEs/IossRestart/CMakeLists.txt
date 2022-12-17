# Name the test with the directory name
get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${parentDirName}_${dirName})

# Copy mesh files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/quadOut.exo
               ${CMAKE_CURRENT_BINARY_DIR}/quadOut.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/quadOut.exo.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/quadOut.exo.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/quadOut.exo.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/quadOut.exo.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/quadOut.exo.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/quadOut.exo.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/quadOut.exo.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/quadOut.exo.4.3 COPYONLY)

####################################
###          Main tests          ###
####################################

set(testName ${testNameRoot})

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
endif ()

####################################
###        Refine tests          ###
####################################

IF(ALBANY_STK_PERCEPT)
  set (testName ${testNameRoot}_Refine)
  if (ALBANY_EPETRA) 
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputRefine1.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputRefine1.yaml COPYONLY)

    add_test(${testName}_1_Epetra ${Albany.exe} inputRefine1.yaml)
    set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Basic;Epetra;Forward")

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputRefine2.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputRefine2.yaml COPYONLY)

    add_test(${testName}_2_Epetra ${Albany.exe} inputRefine2.yaml)
    set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Basic;Epetra;Forward")
  endif()
ENDIF()
