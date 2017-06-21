

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSingleMaterial.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSingleMaterial.yaml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/10_blocks_1.0_size_1.0_meshSize.g
               ${CMAKE_CURRENT_BINARY_DIR}/10_blocks_1.0_size_1.0_meshSize.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMultiMat_10.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMultiMat_10.yaml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_10.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials_10.yaml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/100_blocks_1.0_size_1.0_meshSize.g
               ${CMAKE_CURRENT_BINARY_DIR}/100_blocks_1.0_size_1.0_meshSize.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMultiMat_100.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMultiMat_100.yaml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMultiMat_100T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMultiMat_100T.yaml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_100.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials_100.yaml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName}_SingleMaterial_SERIAL ${SerialAlbany.exe} inputSingleMaterial.yaml)
add_test(${testName}_MultiMaterial_10_SERIAL ${SerialAlbany.exe} inputMultiMat_10.yaml)
add_test(${testName}_MultiMaterial_100_SERIAL ${SerialAlbany.exe} inputMultiMat_100.yaml)
endif()

add_test(${testName}_SingleMaterial_SERIAL ${SerialAlbanyT.exe} inputSingleMaterial.yaml)

add_test(${testName}_MultiMaterial_10_SERIAL ${SerialAlbanyT.exe} inputMultiMat_10.yaml)

add_test(${testName}_MultiMaterial_100_SERIAL ${SerialAlbanyT.exe}
inputMultiMat_100T.yaml)

