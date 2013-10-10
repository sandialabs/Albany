# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMechanicsSimple.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMechanicsSimple.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PoroMaterialsSimple.xml
               ${CMAKE_CURRENT_BINARY_DIR}/PoroMaterialsSimple.xml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}Simple NAME)
# 3. Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY)
  add_test(${testName}_Serial ${SerialAlbany.exe} inputMechanicsSimple.xml)
ENDIF()

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMechanics.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMechanics.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PoroMaterials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/PoroMaterials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface3ElemMesh.g
               ${CMAKE_CURRENT_BINARY_DIR}/surface3ElemMesh.g COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY) 
  add_test(${testName}_Serial ${SerialAlbany.exe} inputMechanics.xml)
ENDIF()

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMechanicsLocalized.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMechanicsLocalized.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PoroMaterialsLocalized.xml
               ${CMAKE_CURRENT_BINARY_DIR}/PoroMaterialsLocalized.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface3ElemMesh.g
               ${CMAKE_CURRENT_BINARY_DIR}/surface3ElemMesh.g COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}Localized NAME)
# 3. Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY)
  add_test(${testName}_Serial ${SerialAlbany.exe} inputMechanicsLocalized.xml)
ENDIF()

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputParallelFlow.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputParallelFlow.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PoroMaterialsLocalized.xml
               ${CMAKE_CURRENT_BINARY_DIR}/PoroMaterialsLocalized.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface3ElemMesh.g
               ${CMAKE_CURRENT_BINARY_DIR}/surface3ElemMesh.g COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}ParallelFlow NAME)
# 3. Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY)
  add_test(${testName}_Serial ${SerialAlbany.exe} inputParallelFlow.xml)
ENDIF()

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputOrthogonalFlow.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputOrthogonalFlow.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PoroMaterialsLocalized.xml
               ${CMAKE_CURRENT_BINARY_DIR}/PoroMaterialsLocalized.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface3ElemMesh.g
               ${CMAKE_CURRENT_BINARY_DIR}/surface3ElemMesh.g COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}OrthogonalFlow NAME)
# 3. Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY)
  add_test(${testName}_Serial ${SerialAlbany.exe} inputOrthogonalFlow.xml)
ENDIF()


