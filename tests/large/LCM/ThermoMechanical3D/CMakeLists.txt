
# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSplitSolution.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSplitSolution.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSplitSolutionT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSplitSolutionT.yaml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
if (ALBANY_IFPACK2)
add_test(${testName} ${Albany.exe} inputT.yaml)
set_tests_properties(${testName} PROPERTIES LABELS "LCM;Tpetra;Forward")
add_test(${testName}_SplitOutput ${Albany.exe} inputSplitSolutionT.yaml)
set_tests_properties(${testName}_SplitOutput PROPERTIES LABELS "LCM;Tpetra;Forward")
endif ()
