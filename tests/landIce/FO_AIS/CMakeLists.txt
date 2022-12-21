
get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${parentDirName}_${dirName}_16km)

# This variable will be set in the input files during the 'configure_file' call
if (ALBANY_PARALELL_EXODUS)
  set (USE_SERIAL_MESH "true")
else ()
  # If there's no Iopx, we *must* have ALBANY_DECOMP defined (to decompose the mesh)
  # OR execute with only one rank.
  IF (ALBANY_MPI AND NOT ${ALBANY_SEACAS_PATH})
    MESSAGE (FATAL_ERROR "Error! Cannot execute FO_AIS tests in parallel without Iopx or decomp from Trilinos.\n")
  ENDIF()
  set (USE_SERIAL_MESH "false")
endif()

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
                 
if (NOT ALBANY_PARALELL_EXODUS AND ALBANY_MPI)
  add_test (NAME ${testNameRoot}_decompMesh
        COMMAND ${SEACAS_DECOMP} -processors ${MPIMNP} antarctica_2d.exo
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties (${testNameRoot}_decompMesh PROPERTIES FIXTURES_SETUP antarcticaMeshSetup)
endif()


if (ALBANY_EPETRA) 
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputML.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputML.yaml)
  add_test(${testNameRoot}_ML ${Albany8.exe} inputML.yaml)
    
  if (NOT ALBANY_PARALELL_EXODUS AND ALBANY_MPI)
    set_tests_properties(${testNameRoot}_ML
                     PROPERTIES
                     LABELS            "LandIce;Epetra;Forward"
                     FIXTURES_REQUIRED antarcticaMeshSetup)
  else()
    set_tests_properties(${testNameRoot}_ML PROPERTIES LABELS "LandIce;Epetra;Forward")
  endif()
endif()

if(ALBANY_MUELU)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMueLu.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputMueLu.yaml)
  add_test(${testNameRoot}_MueLu ${Albany8.exe} inputMueLu.yaml)
    
  if (NOT ALBANY_PARALELL_EXODUS AND ALBANY_MPI)
    set_tests_properties(${testNameRoot}_MueLu
                     PROPERTIES
                     LABELS            "LandIce;Tpetra;Forward"
                     FIXTURES_REQUIRED antarcticaMeshSetup)
  else()
    set_tests_properties(${testNameRoot}_MueLu PROPERTIES LABELS "LandIce;Tpetra;Forward")
  endif()
endif()


# if (ALBANY_FROSCH)
#   configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_FROSch.yaml
#                  ${CMAKE_CURRENT_BINARY_DIR}/input_FROSch.yaml)
#   add_test(${testNameRoot}_FROSch ${Albany8.exe} input_FROSch.yaml)
#   if (NOT ALBANY_PARALELL_EXODUS AND ALBANY_MPI)
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
