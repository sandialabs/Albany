# Name the test with the directory name
get_filename_component(testNameRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)

####################################
###    1D Standing Wave test     ###
####################################

IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_EPETRA)
  set(testName ${testNameRoot}_1D_standingWave)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input1D_standingWave.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input1D_standingWave.yaml COPYONLY)

  add_test(${testName}_Epetra ${SerialAlbany.exe} input1D_standingWave.yaml)
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Demo;Epetra;Forward;Serial")
endif()

####################################
###       2D Steady test         ###
####################################

IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_EPETRA)
  set(testName ${testNameRoot}_2DSteady)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2D.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input2D.yaml COPYONLY)

  add_test(${testName}_Epetra ${SerialAlbany.exe} input2D.yaml)
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Demo;Epetra;Forward;Serial")
endif()

####################################
###      2D Press Pulse test     ###
####################################

if (ALBANY_IFPACK2)
  set(testName ${testNameRoot}_2DUnsteadyInvPressPulse)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2Dunsteady.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input2Dunsteady.yaml COPYONLY)

  add_test(${testName}_Tpetra ${Albany.exe} input2Dunsteady.yaml)
  set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "Demo;Tpetra;Forward")
endif()

####################################
###      2D Euler MMS test       ###
####################################

IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_EPETRA)
  set(testName ${testNameRoot}_2DUnsteadyEulerMMS)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2DunsteadyMMS.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input2DunsteadyMMS.yaml COPYONLY)
  add_test(${testName}_Epetra ${SerialAlbany.exe} input2DunsteadyMMS.yaml)
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Demo;Epetra;Forward;Serial")
ENDIF()

####################################
###     2D Driven Pulse test     ###
####################################

IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_EPETRA)
  set(testName ${testNameRoot}_2DDrivenPulse)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2D_drivenPulse.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input2D_drivenPulse.yaml COPYONLY)

  add_test(${testName}_Epetra ${SerialAlbany.exe} input2D_drivenPulse.yaml)
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Demo;Epetra;Forward;Serial")
ENDIF()

####################################
###    3D Unsteady Euler test    ###
####################################

IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_EPETRA)
  set(testName ${testNameRoot}_3DUnsteadyEuler)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input3Dunsteady.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input3Dunsteady.yaml COPYONLY)

  add_test(${testName}_Epetra ${SerialAlbany.exe} input3Dunsteady.yaml)
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Demo;Epetra;Forward;Serial")
ENDIF()

####################################
###      3D Unsteady NS test     ###
####################################

IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_EPETRA)
  set(testName ${testNameRoot}_3DUnsteadyNS)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input3DunsteadyNS.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input3DunsteadyNS.yaml COPYONLY)

  add_test(${testName}_Epetra ${SerialAlbany.exe} input3DunsteadyNS.yaml)
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Demo;Epetra;Forward;Serial")
ENDIF()
