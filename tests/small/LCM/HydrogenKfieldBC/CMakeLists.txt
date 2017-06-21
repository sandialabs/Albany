# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/K_field.g
               ${CMAKE_CURRENT_BINARY_DIR}/K_field.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/K_field.g.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/K_field.g.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/K_field.g.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/K_field.g.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/K_field.g.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/K_field.g.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/K_field.g.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/K_field.g.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/HMaterials_fast_path.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/HMaterials_fast_path.yaml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
add_test(${testName} ${AlbanyT.exe} input.yaml)
