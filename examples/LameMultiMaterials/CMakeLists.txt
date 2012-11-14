

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSingleMaterial.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSingleMaterial.xml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/10_blocks_1.0_size_1.0_meshSize.g
               ${CMAKE_CURRENT_BINARY_DIR}/10_blocks_1.0_size_1.0_meshSize.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMultiMat_10.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMultiMat_10.xml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_10.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials_10.xml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/100_blocks_1.0_size_1.0_meshSize.g
               ${CMAKE_CURRENT_BINARY_DIR}/100_blocks_1.0_size_1.0_meshSize.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMultiMat_100.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMultiMat_100.xml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_100.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials_100.xml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
add_test(${testName}_SingleMaterial_SERIAL ${SerialAlbany.exe} inputSingleMaterial.xml)

add_test(${testName}_MultiMaterial_10_SERIAL ${SerialAlbany.exe} inputMultiMat_10.xml)
add_test(${testName}_MultiMaterial_100_SERIAL ${SerialAlbany.exe} inputMultiMat_100.xml)

