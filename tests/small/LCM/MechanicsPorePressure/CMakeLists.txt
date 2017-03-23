# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMechanicsSimple.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMechanicsSimple.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMechanicsSimpleT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMechanicsSimpleT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PoroMaterialsSimple.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/PoroMaterialsSimple.yaml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}Simple NAME)
# 3. Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY)
  if (ALBANY_EPETRA) 
    add_test(${testName}_Serial ${SerialAlbany.exe} inputMechanicsSimple.yaml)
  endif()
  if (ALBANY_IFPACK2) 
    add_test(${testName}_Serial_Tpetra ${SerialAlbanyT.exe} inputMechanicsSimpleT.yaml)
  endif()
ENDIF()

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMechanics.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMechanics.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PoroMaterials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/PoroMaterials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface3ElemMesh.g
               ${CMAKE_CURRENT_BINARY_DIR}/surface3ElemMesh.g COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_EPETRA) 
  add_test(${testName}_Serial ${SerialAlbany.exe} inputMechanics.yaml)
ENDIF()

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMechanicsLocalized.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMechanicsLocalized.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMechanicsLocalizedT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMechanicsLocalizedT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PoroMaterialsLocalized.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/PoroMaterialsLocalized.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface3ElemMesh.g
               ${CMAKE_CURRENT_BINARY_DIR}/surface3ElemMesh.g COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}Localized NAME)
# 3. Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY)
  if (ALBANY_EPETRA) 
    add_test(${testName}_Serial ${SerialAlbany.exe} inputMechanicsLocalized.yaml)
  endif()
  if (ALBANY_IFPACK2) 
    add_test(${testName}_Serial_Tpetra ${SerialAlbanyT.exe} inputMechanicsLocalizedT.yaml)
  endif()
ENDIF()

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputParallelFlow.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputParallelFlow.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputParallelFlowT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputParallelFlowT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PoroMaterialsLocalized.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/PoroMaterialsLocalized.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface3ElemMesh.g
               ${CMAKE_CURRENT_BINARY_DIR}/surface3ElemMesh.g COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}ParallelFlow NAME)
# 3. Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY)
  if (ALBANY_EPETRA) 
    add_test(${testName}_Serial ${SerialAlbany.exe} inputParallelFlow.yaml)
  endif()
  if (ALBANY_IFPACK2) 
    add_test(${testName}_Serial_Tpetra ${SerialAlbanyT.exe} inputParallelFlowT.yaml)
  endif()
ENDIF()

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputOrthogonalFlow.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputOrthogonalFlow.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PoroMaterialsLocalized.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/PoroMaterialsLocalized.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface3ElemMesh.g
               ${CMAKE_CURRENT_BINARY_DIR}/surface3ElemMesh.g COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}OrthogonalFlow NAME)
# 3. Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_EPETRA)
  add_test(${testName}_Serial ${SerialAlbany.exe} inputOrthogonalFlow.yaml)
ENDIF()


