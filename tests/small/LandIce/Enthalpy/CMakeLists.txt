get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if (ALBANY_PARALELL_EXODUS)
  set (USE_SERIAL_MESH "true")
else ()
  # If there's no Iopx, we *must* have ALBANY_DECOMP defined (to decompose the mesh)
  # OR execute with only one rank.
  if (ALBANY_MPI AND NOT ${ALBANY_SEACAS_PATH})
    MESSAGE (FATAL_ERROR "Error! Cannot execute FO_GIS tests in parallel without Iopx or decomp from Trilinos.\n")
  endif()
  set (USE_SERIAL_MESH "false")
endif()

# Create the tests
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_enthalpy_humboldt.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_enthalpy_humboldt.yaml)
add_test(${testName}_Humboldt ${Albany.exe} input_enthalpy_humboldt.yaml)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_enthalpy_humboldt_muelu_P1semiR1transP2const.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_enthalpy_humboldt_muelu_P1semiR1transP2const.yaml)
add_test(${testName}_Humboldt_MueLu_P1semiR1transP2const ${Albany.exe} input_enthalpy_humboldt_muelu_P1semiR1transP2const.yaml)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_enthalpy_humboldt_muelu_P1constR1transP2const.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_enthalpy_humboldt_muelu_P1constR1transP2const.yaml)
add_test(${testName}_Humboldt_MueLu_P1constR1transP2const ${Albany.exe} input_enthalpy_humboldt_muelu_P1constR1transP2const.yaml)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_enthalpy_humboldt_muelu_P1linR1transP2const.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_enthalpy_humboldt_muelu_P1linR1transP2const.yaml)
add_test(${testName}_Humboldt_MueLu_P1linR1transP2const ${Albany.exe} input_enthalpy_humboldt_muelu_P1linR1transP2const.yaml)
 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_enthalpy_humboldt_muelu_P1semiR1constP2const.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_enthalpy_humboldt_muelu_P1semiR1constP2const.yaml)
add_test(${testName}_Humboldt_MueLu_P1semiR1constP2const ${Albany.exe} input_enthalpy_humboldt_muelu_P1semiR1constP2const.yaml)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_enthalpy_humboldt_muelu_P1semiR1linP2const.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_enthalpy_humboldt_muelu_P1semiR1linP2const.yaml)
add_test(${testName}_Humboldt_MueLu_P1semiR1linP2const ${Albany.exe} input_enthalpy_humboldt_muelu_P1semiR1linP2const.yaml)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_enthalpy_humboldt_muelu_P1semiR1semiP2const.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_enthalpy_humboldt_muelu_P1semiR1semiP2const.yaml)
add_test(${testName}_Humboldt_MueLu_P1semiR1semiP2const ${Albany.exe} input_enthalpy_humboldt_muelu_P1semiR1semiP2const.yaml)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_enthalpy_humboldt_muelu_P1semiR1transP2pg.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_enthalpy_humboldt_muelu_P1semiR1transP2pg.yaml)
add_test(${testName}_Humboldt_MueLu_P1semiR1transP2pg ${Albany.exe} input_enthalpy_humboldt_muelu_P1semiR1transP2pg.yaml)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_enthalpy_humboldt_muelu_P1semiR1transP2sa.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_enthalpy_humboldt_muelu_P1semiR1transP2sa.yaml)
add_test(${testName}_Humboldt_MueLu_P1semiR1transP2sa ${Albany.exe} input_enthalpy_humboldt_muelu_P1semiR1transP2sa.yaml)

# If Iopx is not present, we must decompose the input mesh first
if (NOT ALBANY_PARALELL_EXODUS AND ALBANY_MPI)
    add_test (NAME FO_GIS_HUMBOLDT_decompMesh
            COMMAND ${SEACAS_DECOMP} -processors ${MPIMNP} humboldt_2d.exo
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../AsciiMeshes/Humboldt)
    set_tests_properties (FO_GIS_HUMBOLDT_decompMesh PROPERTIES FIXTURES_SETUP humboldtMeshSetup)

    set_tests_properties(${testName}_Humboldt
                       PROPERTIES
                       LABELS "LandIce;Tpetra;Forward"
                       FIXTURES_REQUIRED humboldtMeshSetup)
else ()
    set_tests_properties(${testName}_Humboldt
                       PROPERTIES
                       LABELS "LandIce;Tpetra;Forward")
endif()  
  
if (ALBANY_EPETRA)
  # Copy input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_kleiner_B.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_kleiner_B.yaml COPYONLY)

  # Create the test
  add_test(${testName}_Kleiner_B ${SerialAlbany.exe} input_kleiner_B.yaml)
  set_tests_properties(${testName}_Kleiner_B PROPERTIES LABELS "LandIce;Epetra;Forward;Serial")
  
  
  # Copy input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_kleiner_A.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_kleiner_A.yaml COPYONLY)

  # Create the test
  add_test(${testName}_Kleiner_A ${SerialAlbany.exe} input_kleiner_A.yaml)
  set_tests_properties(${testName}_Kleiner_A PROPERTIES LABELS "LandIce;Epetra;Forward;Serial")
endif()

