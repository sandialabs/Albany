# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/3materialsJ2.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/3materialsJ2.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SimpleShearMesh.e
               ${CMAKE_CURRENT_BINARY_DIR}/SimpleShearMesh.e COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
add_test(${testName} ${AlbanyT.exe} input.yaml)

