# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMechanicsSimple.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMechanicsSimple.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PoroMaterialsSimple.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/PoroMaterialsSimple.yaml COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}Simple NAME)

# Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY)
  if (ALBANY_IFPACK2) 
    add_test(${testName}_Serial ${SerialAlbanyT.exe} inputMechanicsSimple.yaml)
  endif()
ENDIF()

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMechanics.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMechanics.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PoroMaterials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/PoroMaterials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface3ElemMesh.g
  ${CMAKE_CURRENT_BINARY_DIR}/surface3ElemMesh.g COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#
# amota: Disable this as LCM will not support Epetra. It needs to be converted
# to Tpetra.
#
# Create the test with this name and standard executable
#IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_EPETRA)
#  add_test(${testName}_Serial ${SerialAlbany.exe} inputMechanics.yaml)
#ENDIF()

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMechanicsLocalized.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMechanicsLocalized.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PoroMaterialsLocalized.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/PoroMaterialsLocalized.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface3ElemMesh.g
  ${CMAKE_CURRENT_BINARY_DIR}/surface3ElemMesh.g COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}Localized NAME)

# Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY)
  if (ALBANY_IFPACK2) 
    add_test(${testName}_Serial ${SerialAlbanyT.exe} inputMechanicsLocalized.yaml)
  endif()
ENDIF()

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputParallelFlow.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputParallelFlow.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PoroMaterialsLocalized.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/PoroMaterialsLocalized.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface3ElemMesh.g
  ${CMAKE_CURRENT_BINARY_DIR}/surface3ElemMesh.g COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}ParallelFlow NAME)

# Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY)
  if (ALBANY_IFPACK2) 
    add_test(${testName}_Serial ${SerialAlbanyT.exe} inputParallelFlow.yaml)
  endif()
ENDIF()

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputOrthogonalFlow.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputOrthogonalFlow.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PoroMaterialsLocalized.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/PoroMaterialsLocalized.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface3ElemMesh.g
  ${CMAKE_CURRENT_BINARY_DIR}/surface3ElemMesh.g COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}OrthogonalFlow NAME)

#
# amota: Disable this as LCM will not support Epetra. It needs to be converted
# to Tpetra.
#
# Create the test with this name and standard executable
#IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_EPETRA)
#  add_test(${testName}_Serial ${SerialAlbany.exe} inputOrthogonalFlow.yaml)
#ENDIF()


