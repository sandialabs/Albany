get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${parentDirName}_${dirName})

# This variable will be set in the input files during the 'configure_file' call
if (ALBANY_PARALELL_EXODUS)
  set (USE_SERIAL_MESH "true")
else ()
  # If there's no Iopx, we *must* have ALBANY_DECOMP defined (to decompose the mesh)
  # OR execute with only one rank.
  IF (NOT ${ALBANY_SEACAS_PATH})
    MESSAGE (FATAL_ERROR "Error! Cannot execute FO_GIS tests in parallel without Iopx or decomp from Trilinos.\n")
  ENDIF()
 set (USE_SERIAL_MESH "false")
endif()


if (ALBANY_AMESOS2_SUPERLUDIST)
  set (DIRECT_SOLVER "superlu_dist")
elseif (ALBANY_AMESOS2_SUPERLU)
  set (DIRECT_SOLVER "superlu")
else ()
  set (DIRECT_SOLVER "klu")
endif()

# Copy input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_FO_Thermo_dry_bed_test.yaml
                ${CMAKE_CURRENT_BINARY_DIR}/input_FO_Thermo_dry_bed_test.yaml)
                
# Copy input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_FO_Thermo_wet_bed_test.yaml
                ${CMAKE_CURRENT_BINARY_DIR}/input_FO_Thermo_wet_bed_test.yaml)
                
# Create the tests
set (testName ${testNameRoot}_Dry_Bed)
add_test(${testName} ${Albany.exe} input_FO_Thermo_dry_bed_test.yaml)
set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward")

set (testName ${testNameRoot}_Wet_Bed)
add_test(${testName} ${Albany.exe} input_FO_Thermo_wet_bed_test.yaml)
set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward")

# If Iopx is not present, we must decompose the input mesh first
if (NOT ALBANY_PARALELL_EXODUS)
  add_test (NAME ${testNameRoot}_Humboldt_decompMesh
          COMMAND ${SerialSeacasDecomp.exe} -processors ${MPIMNP} humboldt_2d.exo
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../AsciiMeshes/Humboldt)
  set_tests_properties (${testNameRoot}_Humboldt_decompMesh PROPERTIES FIXTURES_SETUP humboldtMeshSetup2d)

  add_test (NAME ${testNameRoot}_Humboldt_contiguous_2d_decompMesh
        COMMAND ${SerialSeacasDecomp.exe} -processors ${MPIMNP} humboldt_contiguous_2d.exo
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../AsciiMeshes/Humboldt)
  set_tests_properties (${testNameRoot}_Humboldt_contiguous_2d_decompMesh PROPERTIES FIXTURES_SETUP humboldtMeshSetupContiguous2d)
endif()

# Copy input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_FO_Thermo_wet_bed_test_fea.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_FO_Thermo_wet_bed_test_fea.yaml)

# Create the tests
set (testName ${testNameRoot}_Wet_Bed_FEA)
add_test(${testName} ${Albany.exe} input_FO_Thermo_wet_bed_test_fea.yaml)
set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward")

if (ALBANY_FROSCH)
  # Copy input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_FO_Thermo_wet_bed_test_FROSch.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_FO_Thermo_wet_bed_test_FROSch.yaml)
  
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_FO_Thermo_Humboldt_fluxDiv.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_FO_Thermo_Humboldt_fluxDiv.yaml)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_FO_Thermo_Humboldt_depthInt_fluxDiv.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_FO_Thermo_Humboldt_depthInt_fluxDiv.yaml)
  
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_FO_Thermo_Humboldt_lubricated.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_FO_Thermo_Humboldt_lubricated.yaml)
  
  # Create the tests
  set (testName ${testNameRoot}_Wet_Bed_FROSch)
  add_test(${testName} ${Albany.exe} input_FO_Thermo_wet_bed_test_FROSch.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward;FROSch")
  
  set (testName ${testNameRoot}_Humboldt_fluxDiv)
  add_test(${testName} ${Albany.exe} input_FO_Thermo_Humboldt_fluxDiv.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward;FROSch")

  set (testName ${testNameRoot}_Humboldt_depthIntegrated_fluxDiv)
  add_test(${testName} ${Albany.exe} input_FO_Thermo_Humboldt_depthInt_fluxDiv.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward;FROSch")
  
  set (testName ${testNameRoot}_Humboldt_lubricated)
  add_test(${testName} ${Albany.exe} input_FO_Thermo_Humboldt_lubricated.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward;FROSch")

  if (NOT ALBANY_PARALELL_EXODUS)
    set_tests_properties(${testNameRoot}_Humboldt_fluxDiv
      PROPERTIES FIXTURES_REQUIRED humboldtMeshSetup2d)

    set_tests_properties(${testNameRoot}_Humboldt_depthIntegrated_fluxDiv
      PROPERTIES FIXTURES_REQUIRED humboldtMeshSetup2d)

    set_tests_properties(${testNameRoot}_Humboldt_lubricated
      PROPERTIES FIXTURES_REQUIRED humboldtMeshSetup2d)
  endif()
endif()

if (ALBANY_TEKO AND ALBANY_MUELU AND ALBANY_FROSCH)
  # Copy input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_FO_Thermo_wet_bed_test_teko.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_FO_Thermo_wet_bed_test_teko.yaml)

  # Create the tests
  set (testName ${testNameRoot}_Wet_Bed_Teko)
  add_test(${testName} ${Albany.exe} input_FO_Thermo_wet_bed_test_teko.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward")
endif()

if(${PYTHON_TEST})
if (ALBANY_FROSCH AND ALBNAY_TEKO)
  set(TESTFILES hessian_comparison.py humboldt_analysis.yaml humboldt_analysis_contiguous.yaml H-ref-000.mm H-ref-000_contiguous.mm ${PYTHON_TEST_HELPERS_DIR}/__init__.py ${PYTHON_TEST_HELPERS_DIR}/matrix_reader.py)
  file(COPY ${TESTFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

  set (testName_Contiguous ${testNameRoot}_Contiguous_Mesh_Hessian)
  add_test(${testName_Contiguous}_Serial "${PYTHON_EXECUTABLE}" "hessian_comparison.py" "${SerialAlbanyAnalysis.exe}" "humboldt_analysis_contiguous.yaml" "H-ref-000_contiguous.mm")
  set_tests_properties(${testName_Contiguous}_Serial PROPERTIES LABELS "LandIce;Hessian")

  set (testName_Non_Contiguous ${testNameRoot}_Non_Contiguous_Mesh_Hessian)
  add_test(${testName_Non_Contiguous}_Serial "${PYTHON_EXECUTABLE}" "hessian_comparison.py" "${SerialAlbanyAnalysis.exe}" "humboldt_analysis.yaml" "H-ref-000.mm")
  set_tests_properties(${testName_Non_Contiguous}_Serial PROPERTIES LABELS "LandIce;Hessian")
  set_tests_properties(${testName_Non_Contiguous}_Serial PROPERTIES DEPENDS  ${testName_Contiguous}_Serial)

  add_test(${testName_Contiguous}_Parallel "${PYTHON_EXECUTABLE}" "hessian_comparison.py" "${AlbanyAnalysis.exe}"  "humboldt_analysis_contiguous.yaml" "H-ref-000_contiguous.mm")
  set_tests_properties(${testName_Contiguous}_Parallel PROPERTIES LABELS "LandIce;Hessian")
  # Prevent the two tests to be run at the same time as they rely on written files:
  set_tests_properties(${testName_Contiguous}_Parallel PROPERTIES DEPENDS  ${testName_Non_Contiguous}_Serial)
  
  add_test(${testName_Non_Contiguous}_Parallel "${PYTHON_EXECUTABLE}" "hessian_comparison.py" "${AlbanyAnalysis.exe}"  "humboldt_analysis.yaml" "H-ref-000.mm")
  set_tests_properties(${testName_Non_Contiguous}_Parallel PROPERTIES LABELS "LandIce;Hessian")
  # Prevent the two tests to be run at the same time as they rely on written files:
  set_tests_properties(${testName_Non_Contiguous}_Parallel PROPERTIES DEPENDS  ${testName_Contiguous}_Parallel)

  if (NOT ALBANY_PARALELL_EXODUS)
    set_tests_properties(${testNameRoot}_Contiguous_Mesh_Hessian_Parallel
      PROPERTIES FIXTURES_REQUIRED humboldtMeshSetup2d)

    set_tests_properties(${testNameRoot}_Non_Contiguous_Mesh_Hessian_Parallel
      PROPERTIES FIXTURES_REQUIRED humboldtMeshSetupContiguous2d)
  endif()
endif()
endif()
