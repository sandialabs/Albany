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
add_test(${testName}_tetra4 ${SerialAlbany.exe} input_tetra4.yaml)
set_tests_properties(${testName}_tetra4 PROPERTIES LABELS "LCM;Tpetra;Forward")
add_test(${testName}_tetra10 ${SerialAlbany.exe} input_tetra10.yaml)
set_tests_properties(${testName}_tetra10 PROPERTIES LABELS "LCM;Tpetra;Forward")

add_test(${testName}_tetra4_tip ${SerialAlbany.exe} input_tetra4_tip.yaml)
set_tests_properties(${testName}_tetra4_tip PROPERTIES LABELS "LCM;Tpetra;Forward")
add_test(${testName}_tetra10_tip ${SerialAlbany.exe} input_tetra10_tip.yaml)
set_tests_properties(${testName}_tetra10_tip PROPERTIES LABELS "LCM;Tpetra;Forward")

add_test(${testName}_tetra4_trac ${SerialAlbany.exe} input_tetra4_trac.yaml)
set_tests_properties(${testName}_tetra4_trac PROPERTIES LABELS "LCM;Tpetra;Forward")
add_test(${testName}_tetra10_trac ${SerialAlbany.exe} input_tetra10_trac.yaml)
set_tests_properties(${testName}_tetra10_trac PROPERTIES LABELS "LCM;Tpetra;Forward")

IF (ALBANY_IOPX)
add_test(${testName}_hex8 ${Albany.exe} input_hex8.yaml)
set_tests_properties(${testName}_hex8 PROPERTIES LABELS "LCM;Tpetra;Forward")
add_test(${testName}_hex8_tip ${Albany.exe} input_hex8_tip.yaml)
set_tests_properties(${testName}_hex8_tip PROPERTIES LABELS "LCM;Tpetra;Forward")
add_test(${testName}_hex8_trac ${Albany.exe} input_hex8_trac.yaml)
set_tests_properties(${testName}_hex8_trac PROPERTIES LABELS "LCM;Tpetra;Forward")
endif ()

endif ()
