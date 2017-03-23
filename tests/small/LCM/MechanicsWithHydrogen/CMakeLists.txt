

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SimpleMaterials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/SimpleMaterials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2hex.g
               ${CMAKE_CURRENT_BINARY_DIR}/2hex.g COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
# FIXME - something is just wrong here.
# IF(NOT ALBANY_PARALLEL_ONLY)
#   if (ALBANY_EPETRA) 
#     add_test(${testName}_SERIAL ${SerialAlbany.exe})
#   endif()
#   if (ALBANY_IFPACK2) 
#     add_test(${testName}_SERIAL_Tpetra ${SerialAlbanyT.exe} inputT.yaml)
#   endif()
# ENDIF() 

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputParallel.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputParallel.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputParallelT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputParallelT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/HMaterials_fast_path.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/HMaterials_fast_path.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface3ElemMesh.g
               ${CMAKE_CURRENT_BINARY_DIR}/surface3ElemMesh.g COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}Parallel NAME)
# 3. Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY)
  if (ALBANY_EPETRA) 
    add_test(${testName}_SERIAL ${SerialAlbany.exe} inputParallel.yaml)
  endif()
  if (ALBANY_IFPACK2) 
    add_test(${testName}_SERIAL_Tpetra ${SerialAlbanyT.exe} inputParallelT.yaml)
  endif()
ENDIF()

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputOrthogonal.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputOrthogonal.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputOrthogonalT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputOrthogonalT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/HMaterials_fast_path.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/HMaterials_fast_path.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface3ElemMesh.g
               ${CMAKE_CURRENT_BINARY_DIR}/surface3ElemMesh.g COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}Orthogonal NAME)
# 3. Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY)
  if (ALBANY_EPETRA) 
    add_test(${testName}_SERIAL ${SerialAlbany.exe} inputOrthogonal.yaml)
  endif()
  if (ALBANY_IFPACK2) 
    add_test(${testName}_SERIAL_Tpetra ${SerialAlbanyT.exe} inputOrthogonalT.yaml)
  endif() 
ENDIF()
