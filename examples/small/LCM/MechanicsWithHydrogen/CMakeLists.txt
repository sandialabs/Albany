

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SimpleMaterials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SimpleMaterials.xml COPYONLY)
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
#     add_test(${testName}_SERIAL_Tpetra ${SerialAlbanyT.exe} inputT.xml)
#   endif()
# ENDIF() 

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputParallel.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputParallel.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputParallelT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputParallelT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/HMaterials_fast_path.xml
               ${CMAKE_CURRENT_BINARY_DIR}/HMaterials_fast_path.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface3ElemMesh.g
               ${CMAKE_CURRENT_BINARY_DIR}/surface3ElemMesh.g COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}Parallel NAME)
# 3. Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY)
  if (ALBANY_EPETRA) 
    add_test(${testName}_SERIAL ${SerialAlbany.exe} inputParallel.xml)
  endif()
  if (ALBANY_IFPACK2) 
    add_test(${testName}_SERIAL_Tpetra ${SerialAlbanyT.exe} inputParallelT.xml)
  endif()
ENDIF()

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputOrthogonal.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputOrthogonal.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputOrthogonalT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputOrthogonalT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/HMaterials_fast_path.xml
               ${CMAKE_CURRENT_BINARY_DIR}/HMaterials_fast_path.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface3ElemMesh.g
               ${CMAKE_CURRENT_BINARY_DIR}/surface3ElemMesh.g COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}Orthogonal NAME)
# 3. Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY)
  if (ALBANY_EPETRA) 
    add_test(${testName}_SERIAL ${SerialAlbany.exe} inputOrthogonal.xml)
  endif()
  if (ALBANY_IFPACK2) 
    add_test(${testName}_SERIAL_Tpetra ${SerialAlbanyT.exe} inputOrthogonalT.xml)
  endif() 
ENDIF()
