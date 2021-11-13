get_filename_component(testNameRoot ${CMAKE_CURRENT_SOURCE_DIR}_16km NAME)

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
                 
if (NOT ALBANY_IOPX AND ALBANY_MPI)
  add_test (NAME FO_AIS_decompMesh
        COMMAND ${SEACAS_DECOMP} -processors ${MPIMNP} antarctica_2d.exo
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties (FO_AIS_decompMesh PROPERTIES FIXTURES_SETUP antarcticaMeshSetup)
endif()


if (ALBANY_EPETRA) 
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputML.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputML.yaml COPYONLY)
  add_test(${testNameRoot}_ML ${Albany8.exe} inputML.yaml)
    
  if (NOT ALBANY_IOPX AND ALBANY_MPI)
    set_tests_properties(${testNameRoot}_ML
                     PROPERTIES
                     LABELS            "LandIce;Epetra;Forward"
                     FIXTURES_REQUIRED antarcticaMeshSetup)
  else()
    set_tests_properties(${testNameRoot}_ML PROPERTIES LABELS "LandIce;Epetra;Forward")
  endif()
endif()

if(ALBANY_MUELU)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMueLuShort.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputMueLuShort.yaml COPYONLY)
  add_test(${testNameRoot}_MueLuShortList ${Albany8.exe} inputMueLuShort.yaml)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMueLu.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputMueLu.yaml COPYONLY)
  add_test(${testNameRoot}_MueLu ${Albany8.exe} inputMueLu.yaml)
    
  if (NOT ALBANY_IOPX AND ALBANY_MPI)
    set_tests_properties(${testNameRoot}_MueLuShortList
                     PROPERTIES
                     LABELS            "LandIce;Tpetra;Forward"
                     FIXTURES_REQUIRED antarcticaMeshSetup)
    set_tests_properties(${testNameRoot}_MueLu
                     PROPERTIES
                     LABELS            "LandIce;Tpetra;Forward"
                     FIXTURES_REQUIRED antarcticaMeshSetup)
  else()
    set_tests_properties(${testNameRoot}_MueLuShortList PROPERTIES LABELS "LandIce;Tpetra;Forward")
    set_tests_properties(${testNameRoot}_MueLu PROPERTIES LABELS "LandIce;Tpetra;Forward")
  endif()
endif()


# if (ALBANY_FROSCH)
#   configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_FROSch.yaml
#                  ${CMAKE_CURRENT_BINARY_DIR}/input_FROSch.yaml COPYONLY)
#   add_test(${testNameRoot}_FROSch ${Albany8.exe} input_FROSch.yaml)
#   if (NOT ALBANY_IOPX AND ALBANY_MPI)
#     set_tests_properties(${testNameRoot}_FROSch
#                      PROPERTIES
#                      LABELS            "LandIce;Tpetra;Forward"
#                      FIXTURES_REQUIRED antarcticaMeshSetup)
#   else()
#     set_tests_properties(${testNameRoot}_FROSch
#                        PROPERTIES
#                        LABELS "LandIce;Tpetra;Forward")
#   endif()
# endif()=
