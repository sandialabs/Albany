# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tetra10.exo
               ${CMAKE_CURRENT_BINARY_DIR}/tetra10.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tetra4.exo
               ${CMAKE_CURRENT_BINARY_DIR}/tetra4.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hex8.exo
               ${CMAKE_CURRENT_BINARY_DIR}/hex8.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_tetra10_tip.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_tetra10_tip.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_tetra4_tip.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_tetra4_tip.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_hex8_tip.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_hex8_tip.yaml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_tetra10_trac.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_tetra10_trac.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_tetra4_trac.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_tetra4_trac.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_hex8_trac.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_hex8_trac.yaml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_tetra10.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_tetra10.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_tetra4.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_tetra4.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_hex8.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_hex8.yaml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_IFPACK2)
add_test(${testName}_tetra4 ${SerialAlbanyT.exe} input_tetra4.yaml)
add_test(${testName}_tetra10 ${SerialAlbanyT.exe} input_tetra10.yaml)

add_test(${testName}_tetra4_tip ${SerialAlbanyT.exe} input_tetra4_tip.yaml)
add_test(${testName}_tetra10_tip ${SerialAlbanyT.exe} input_tetra10_tip.yaml)

add_test(${testName}_tetra4_trac ${SerialAlbanyT.exe} input_tetra4_trac.yaml)
add_test(${testName}_tetra10_trac ${SerialAlbanyT.exe} input_tetra10_trac.yaml)

IF (ALBANY_IOPX)
add_test(${testName}_hex8 ${AlbanyT.exe} input_hex8.yaml)
add_test(${testName}_hex8_tip ${AlbanyT.exe} input_hex8_tip.yaml)
add_test(${testName}_hex8_trac ${AlbanyT.exe} input_hex8_trac.yaml)
endif ()

endif ()
