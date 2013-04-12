

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/3materialsJ2.xml
               ${CMAKE_CURRENT_BINARY_DIR}/3materialsJ2.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SimpleShearMesh.e
               ${CMAKE_CURRENT_BINARY_DIR}/SimpleShearMesh.e COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
add_test(${testName} ${Albany.exe})

