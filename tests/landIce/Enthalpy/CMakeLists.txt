get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testName ${parentDirName}_${dirName})

if (ALBANY_PARALLEL_EXODUS)
  set (USE_SERIAL_MESH "true")
else ()
  # If there's no Iopx, we *must* have ALBANY_DECOMP defined (to decompose the mesh)
  # OR execute with only one rank.
  IF (NOT SEACAS_DECOMP)
    MESSAGE (FATAL_ERROR "Error! Cannot execute FO_GIS tests in parallel without Iopx or decomp from Trilinos.\n")
  ENDIF()
 set (USE_SERIAL_MESH "false")
endif()


  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_enthalpy_humboldt.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_enthalpy_humboldt.yaml)
 
# Create the test
add_test(${testName}_Humboldt ${Albany.exe} input_enthalpy_humboldt.yaml)
albany_tests_require_static_fad(${testName}_Humboldt)
# If Iopx is not present, the input mesh is decomposed in AsciiMeshes/Humboldt
if (NOT ALBANY_PARALLEL_EXODUS)
    set_tests_properties(${testName}_Humboldt
                       PROPERTIES
                       LABELS "LandIce;Forward"
                       FIXTURES_REQUIRED humboldtMeshSetup2d)
else ()
    set_tests_properties(${testName}_Humboldt
                       PROPERTIES
                       LABELS "LandIce;Forward")
endif()  
  

# Copy input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_kleiner_B.yaml
                ${CMAKE_CURRENT_BINARY_DIR}/input_kleiner_B.yaml COPYONLY)

# Create the test
add_test(${testName}_Kleiner_B ${SerialAlbany.exe} input_kleiner_B.yaml)
set_tests_properties(${testName}_Kleiner_B PROPERTIES LABELS "LandIce;Forward;Serial")


# Copy input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_kleiner_A.yaml
                ${CMAKE_CURRENT_BINARY_DIR}/input_kleiner_A.yaml COPYONLY)

# Create the test
add_test(${testName}_Kleiner_A ${SerialAlbany.exe} input_kleiner_A.yaml)
set_tests_properties(${testName}_Kleiner_A PROPERTIES LABELS "LandIce;Forward;Serial")

