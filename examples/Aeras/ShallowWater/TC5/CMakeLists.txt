 

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_implicitTimeInt.xml
#               ${CMAKE_CURRENT_BINARY_DIR}/input_implicitTimeInt.xml COPYONLY)
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicitTimeInt.xml
#               ${CMAKE_CURRENT_BINARY_DIR}/input_explicitTimeInt.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../grids/QUAD9/uniform_16_quad9.g
               ${CMAKE_CURRENT_BINARY_DIR}/uniform_16_quad9.g COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
add_test(${testName}_Serial ${SerialAlbany.exe})

