# Name the test with the directory name
get_filename_component(testNameRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# This variable will be set in the input files during the 'configure_file' call
if (ALBANY_IOPX)
  set (USE_SERIAL_MESH "true")
#IKT 5/22/19 - commenting out for now as this is leading to compilation error
else ()
  # If there's no Iopx, we *must* have ALBANY_DECOMP defined (to decompose the mesh)
  # OR execute with only one rank.
  IF (ALBANY_MPI AND NOT ${ALBANY_SEACAS_PATH})
    MESSAGE (FATAL_ERROR "Error! Cannot execute FO_GIS tests in parallel without Iopx or decomp from Trilinos.\n")
  ENDIF()
 set (USE_SERIAL_MESH "false")
endif()

####################################
###        Unstruct tests        ###
####################################

set (testName ${testNameRoot}_Steady)

# if (ALBANY_EPETRA)
#   configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis_unstruct_steady.yaml
#                  ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis_unstruct_steady.yaml)
#   add_test(${testName}_Epetra ${Albany.exe}  input_fo_gis_unstruct_steady.yaml)
#   set_tests_properties(${testName}_Epetra
#                        PROPERTIES
#                        LABELS            "LandIce;Epetra;Forward")
# endif()

if (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_dome_steadyT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_dome_steadyT.yaml)
  add_test(${testName}_Dome_Tpetra ${Albany.exe}  input_dome_steadyT.yaml)
  set_tests_properties(${testName}_Dome_Tpetra
                       PROPERTIES
                       LABELS "LandIce;Tpetra;Forward")
endif()
