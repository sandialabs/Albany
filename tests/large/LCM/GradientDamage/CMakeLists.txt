

if (ALBANY_IFPACK2)
# 1'. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)
# 2'. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3'. Create the test with this name and standard executable
add_test(${testName} ${Albany.exe} inputT.yaml)
set_tests_properties(${testName} PROPERTIES LABELS "LCM;Tpetra;Forward")
endif ()

