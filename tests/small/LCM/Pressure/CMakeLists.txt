# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_tetra10.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_tetra10.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tetra10.exo
               ${CMAKE_CURRENT_BINARY_DIR}/tetra10.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_tetra4.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_tetra4.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tetra4.exo
               ${CMAKE_CURRENT_BINARY_DIR}/tetra4.exo COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_IFPACK2)
add_test(${testName}_tetra10 ${AlbanyT.exe} input_tetra10.yaml)
add_test(${testName}_tetra4 ${AlbanyT.exe} input_tetra4.yaml)
endif ()
