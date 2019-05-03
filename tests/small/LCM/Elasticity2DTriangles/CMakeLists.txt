# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_RegressFail.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_RegressFail.yaml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_IFPACK2)
  add_test(${testName} ${Albany.exe} input.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LCM;Tpetra;Forward")
  add_test(${testName}_RegressFail ${Albany.exe} input_RegressFail.yaml)
  set_tests_properties(${testName}_RegressFail PROPERTIES LABELS "LCM;Tpetra;Forward" WILL_FAIL TRUE)
endif ()
