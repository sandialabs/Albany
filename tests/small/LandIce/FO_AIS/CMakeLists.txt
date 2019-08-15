if (ALBANY_IOPX)
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_16km NAME)

  # Copy mesh and input files from source dir to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/antarctica_2d.exo
                 ${CMAKE_CURRENT_BINARY_DIR}/antarctica_2d.exo COPYONLY)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface_height.ascii
                 ${CMAKE_CURRENT_BINARY_DIR}/surface_height.ascii COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thickness.ascii
                 ${CMAKE_CURRENT_BINARY_DIR}/thickness.ascii COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/temperature.ascii
                 ${CMAKE_CURRENT_BINARY_DIR}/temperature.ascii COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/basal_friction_reg.ascii
                 ${CMAKE_CURRENT_BINARY_DIR}/basal_friction_reg.ascii COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface_velocity.ascii
                 ${CMAKE_CURRENT_BINARY_DIR}/surface_velocity.ascii COPYONLY)

  if (ALBANY_EPETRA) 
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputML3.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputML3.yaml COPYONLY)

    add_test(${testName}_ML ${Albany8.exe} inputML3.yaml)
    set_tests_properties(${testName}_ML PROPERTIES LABELS "LandIce;Epetra;Forward")
  endif()

  if(ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMueLuShort3.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputMueLuShort3.yaml COPYONLY)

    add_test(${testName}_MueLu ${Albany8.exe} inputMueLuShort3.yaml)
    set_tests_properties(${testName}_MueLu PROPERTIES LABELS "LandIce;Tpetra;Forward")
  endif()

  if (ALBANY_FROSCH)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input3_FROSch.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input3_FROSch.yaml COPYONLY)
    add_test(${testName}_FROSch_3 ${Albany8.exe} input3_FROSch.yaml)
    set_tests_properties(${testName}_FROSch_3
                         PROPERTIES
                         LABELS "LandIce;Tpetra;Forward")

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputShort_FROSch.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputShort_FROSch.yaml COPYONLY)
    add_test(${testName}_FROSch_short ${Albany8.exe} inputShort_FROSch.yaml)
    set_tests_properties(${testName}_FROSch_short
                         PROPERTIES
                         LABELS "LandIce;Tpetra;Forward")
  endif()
endif()
