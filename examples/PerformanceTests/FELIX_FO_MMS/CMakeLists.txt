# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/data.perf
               ${CMAKE_CURRENT_BINARY_DIR}/data.perf COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
add_test(${testName}_perf ${performanceTestScript})

# Disable test if there isn't an entry for the current machine in data.perf

# Ignore empty tokens in "listification" of strings
CMAKE_POLICY(SET CMP0007 OLD)

FILE(READ data.perf contents)

STRING(REGEX REPLACE ";" "\\\\;" contents "${contents}")
STRING(REGEX REPLACE "\n" ";" contents "${contents}")

FOREACH(text_line ${contents})
#  message("${text_line}")
  STRING(REGEX REPLACE "   " ";" text_line "${text_line}")
  STRING(REGEX REPLACE "  " ";" text_line "${text_line}")
  STRING(REGEX REPLACE " " ";" text_line "${text_line}")
  LIST(GET text_line 0 output)
  IF("${machineName}" STREQUAL output)
    RETURN()
  ENDIF()
ENDFOREACH()
set_tests_properties(${testName}_perf  PROPERTIES REQUIRED_FILES "machine ${machineName} not found in data.perf file")
