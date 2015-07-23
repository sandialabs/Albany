 

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_hv_T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_hv_T.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicit_hv_T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_explicit_hv_T.xml COPYONLY)
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_implicitTimeInt.xml
#               ${CMAKE_CURRENT_BINARY_DIR}/input_implicitTimeInt.xml COPYONLY)
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicitTimeInt.xml
#               ${CMAKE_CURRENT_BINARY_DIR}/input_explicitTimeInt.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../grids/QUAD9/uniform_16_quad9.g
               ${CMAKE_CURRENT_BINARY_DIR}/uniform_16_quad9.g COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(Aeras_${testName} ${Albany.exe})
set_tests_properties(Aeras_${testName}  PROPERTIES LABELS CUDA_TEST)
endif() 
if (NOT ALBANY_MOR)
add_test(Aeras_${testName}_Tpetra ${AlbanyT.exe} inputT.xml)
add_test(Aeras_${testName}_HV_Tpetra ${AlbanyT.exe} input_hv_T.xml)
endif()
