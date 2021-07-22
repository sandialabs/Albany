  set(testName LandIce_FO_Dome)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/circle.msh
                 ${CMAKE_CURRENT_BINARY_DIR}/circle.msh COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/circle_fine.msh
                 ${CMAKE_CURRENT_BINARY_DIR}/circle_fine.msh COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/xyz
                 ${CMAKE_CURRENT_BINARY_DIR}/xyz COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/eles
                 ${CMAKE_CURRENT_BINARY_DIR}/eles COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/sh
                 ${CMAKE_CURRENT_BINARY_DIR}/sh COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/bf
                 ${CMAKE_CURRENT_BINARY_DIR}/bf COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/flwa
                 ${CMAKE_CURRENT_BINARY_DIR}/flwa COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/beta
                 ${CMAKE_CURRENT_BINARY_DIR}/beta COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_dome_ascii.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_dome_ascii.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_dome_restart.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_dome_restart.yaml COPYONLY)

IF (ALBANY_EPETRA) 
  add_test(${testName}_Ascii ${SerialAlbany.exe} input_fo_dome_ascii.yaml)
  set_tests_properties(${testName}_Ascii PROPERTIES LABELS "LandIce;Epetra;Forward;Serial")

  IF (ALBANY_SEACAS)
    add_test(${testName}_Restart ${SerialAlbany.exe} input_fo_dome_restart.yaml)
    set_tests_properties(${testName}_Restart PROPERTIES DEPENDS ${testName}_Ascii_Epetra)
    set_tests_properties(${testName}_Restart PROPERTIES RUN_SERIAL TRUE LABELS "LandIce;Epetra;Forward;Serial")
  ENDIF()
ENDIF()
