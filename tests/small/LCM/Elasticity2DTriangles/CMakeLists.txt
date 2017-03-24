# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_RegressFail.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_RegressFail.yaml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_IFPACK2)
  add_test(${testName} ${AlbanyT.exe} input.yaml)
  add_test(${testName}_RegressFail ${AlbanyT.exe} input_RegressFail.yaml)
  set_tests_properties(${testName}_RegressFail PROPERTIES WILL_FAIL TRUE)
endif ()
