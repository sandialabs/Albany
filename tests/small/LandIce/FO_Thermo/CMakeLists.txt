get_filename_component(testNameRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if (ALBANY_PARALELL_EXODUS)
  set (USE_SERIAL_MESH "true")
else ()
  # If there's no Iopx, we *must* have ALBANY_DECOMP defined (to decompose the mesh)
  # OR execute with only one rank.
  IF (ALBANY_MPI AND NOT ${ALBANY_SEACAS_PATH})
    MESSAGE (FATAL_ERROR "Error! Cannot execute FO_GIS tests in parallel without Iopx or decomp from Trilinos.\n")
  ENDIF()
 set (USE_SERIAL_MESH "false")
endif()

if (ALBANY_EPETRA)

  # Copy input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_FO_Thermo_dry_bed_test.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_FO_Thermo_dry_bed_test.yaml)
                 
  # Copy input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_FO_Thermo_wet_bed_test.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_FO_Thermo_wet_bed_test.yaml)
                 
  # Copy input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_FO_Thermo_Humboldt_fluxDiv.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_FO_Thermo_Humboldt_fluxDiv.yaml)

  # Create the tests
  set (testName ${testNameRoot}_Dry_Bed)
  add_test(${testName} ${Albany.exe} input_FO_Thermo_dry_bed_test.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Epetra;Forward")
  
  set (testName ${testNameRoot}_Wet_Bed)
  add_test(${testName} ${Albany.exe} input_FO_Thermo_wet_bed_test.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Epetra;Forward")
  
  set (testName ${testNameRoot}_Humboldt_fluxDiv)
  add_test(${testName} ${Albany.exe} input_FO_Thermo_Humboldt_fluxDiv.yaml)
  # If Iopx is not present, we must decompose the input mesh first
  if (NOT ALBANY_PARALELL_EXODUS AND ALBANY_MPI)
    add_test (NAME FO_GIS_HUMBOLDT_decompMesh
            COMMAND ${SEACAS_DECOMP} -processors ${MPIMNP} humboldt_2d.exo
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../AsciiMeshes/Humboldt)
    set_tests_properties (FO_GIS_HUMBOLDT_decompMesh PROPERTIES FIXTURES_SETUP humboldtMeshSetup)

    set_tests_properties(${testName}
                       PROPERTIES
                       LABELS            "LandIce;Epetra;Forward"
                       FIXTURES_REQUIRED humboldtMeshSetup)
    else ()
    set_tests_properties(${testName}
                       PROPERTIES
                       LABELS         "LandIce;Epetra;Forward")
  endif()
endif()

# Copy input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_FO_Thermo_wet_bed_test_fea.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_FO_Thermo_wet_bed_test_fea.yaml)

# Create the tests
set (testName ${testNameRoot}_Wet_Bed_FEA)
add_test(${testName} ${Albany.exe} input_FO_Thermo_wet_bed_test_fea.yaml)
set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Tpetra;Forward")

if (ALBANY_FROSCH)
  # Copy input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_FO_Thermo_wet_bed_test_FROSch.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_FO_Thermo_wet_bed_test_FROSch.yaml)

  # Create the tests
  set (testName ${testNameRoot}_Wet_Bed_FROSch)
  add_test(${testName} ${Albany.exe} input_FO_Thermo_wet_bed_test_FROSch.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Tpetra;Forward;FROSch")
endif()

if(${PYTHON_TEST})
if (ALBANY_FROSCH)
  set(TESTFILES hessian_comparison.py humboldt_analysis.yaml H-ref-000.mm ${PYTHON_TEST_HELPERS_DIR}/__init__.py ${PYTHON_TEST_HELPERS_DIR}/matrix_reader.py)
  file(COPY ${TESTFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

  set (testName ${testNameRoot}_Contiguous_Mesh_Hessian)
  add_test(${testName}_Serial "${PYTHON_EXECUTABLE}" "hessian_comparison.py" "${SerialAlbanyAnalysis.exe}" "humboldt_analysis.yaml")
  set_tests_properties(${testName}_Serial PROPERTIES LABELS "LandIce;Tpetra;Hessian")

  if(ALBANY_MPI)
    add_test(${testName}_Parallel "${PYTHON_EXECUTABLE}" "hessian_comparison.py" "${AlbanyAnalysis.exe}"  "humboldt_analysis.yaml")
    set_tests_properties(${testName}_Parallel PROPERTIES LABELS "LandIce;Tpetra;Hessian")
    # Prevent the two tests to be run at the same time as they rely on written files:
    set_tests_properties(${testName}_Parallel PROPERTIES DEPENDS  ${testName}_Serial)
  endif()
endif()
endif()
