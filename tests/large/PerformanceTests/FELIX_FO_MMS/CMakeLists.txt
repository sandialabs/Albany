# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/data.perf
               ${CMAKE_CURRENT_BINARY_DIR}/data.perf COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
add_test(${testName}_perf ${performanceTestScript})
add_test(${testName}_perf_2 ${performanceTestScript_2})

# Disable test if there isn't an entry for the current machine in data.perf

FILE(READ data.perf contents)

STRING(FIND "${contents}" "${machineName}" machinematched)

IF(${machinematched} EQUAL -1)
  set_tests_properties(${testName}_perf  PROPERTIES REQUIRED_FILES "machine ${machineName} not found in data.perf file")
  set_tests_properties(${testName}_perf_2  PROPERTIES REQUIRED_FILES "machine ${machineName}_2 not found in data.perf file")
ENDIF()
