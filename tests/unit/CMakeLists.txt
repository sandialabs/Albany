##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Utility to create unit test
#  Arguments
#    NAME <name>        Name of executable.
#    PREFIX <p>         String to prepend to test name
#    SOURCES <files>    List of files to build the exec
#    LIBS <libs>        List of libs to link to the exec.
#    SERIAL_ONLY        Whether we need to run the test only in serial mode (default: OFF)
#
# If prefix is present, test names start with ${PREFIX}_${NAME},
# otherwise they simply start with ${NAME}.
# Note: the main albanyLib library is already linked, via the albany_ut_main, so no need to specify it

macro (create_unit_test)
  # Parse args
  set (options SERIAL_ONLY)
  set (oneValueArgs NAME PREFIX)
  set (multiValueArgs SOURCES LIBS)
  cmake_parse_arguments (PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if (PARSE_SERIAL_ONLY)
    set (PARALLEL OFF)
  else()
    set (PARALLEL ON)
  endif()

  set (EXEC_NAME ${PARSE_NAME})

  if (PARSE_PREFIX)
    set (TEST_PREFIX ${PARSE_PREFIX}_${PARSE_NAME})
  else()
    set (TEST_PREFIX ${PARSE_NAME})
  endif()

  # Create executable
  add_executable (${EXEC_NAME} ${PARSE_SOURCES})
  list (APPEND PARSE_LIBS albany_ut_main)
  target_link_libraries (${EXEC_NAME} PUBLIC ${PARSE_LIBS})

  # Add tests
  if (PARALLEL)
    add_test (NAME ${TEST_PREFIX}_UnitTest_Serial
              COMMAND ${SERIAL_CALL} ${CMAKE_CURRENT_BINARY_DIR}/${EXEC_NAME})
    add_test (NAME ${TEST_PREFIX}_UnitTest_Parallel
              COMMAND ${PARALLEL_CALL} ${CMAKE_CURRENT_BINARY_DIR}/${EXEC_NAME})
    set_tests_properties (
      ${TEST_PREFIX}_UnitTest_Serial ${TEST_PREFIX}_UnitTest_Parallel
      PROPERTIES LABELS "unit")
  else()
    add_test (NAME ${TEST_PREFIX}_UnitTest
              COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${EXEC_NAME})
    set_tests_properties (
      ${TEST_PREFIX}_UnitTest
      PROPERTIES LABELS "unit")
  endif()
endmacro()

#######################################
#          Unit tests main            #
#######################################

# We build the main as a library, then link it to each test executable
add_library (albany_ut_main Albany_UnitTestMain.cpp)
target_include_directories (albany_ut_main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (albany_ut_main PUBLIC albanyLib)

#######################################
#       Individual unit tests         #
#######################################

get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${dirName})

# NullSpaceUtils unit tests
create_unit_test (NAME    NullSpaceUtils
                  PREFIX  ${dirName}
                  SOURCES NullSpaceUtils.cpp)

# StringUtils unit tests (no MPI test, since it's pointless)
create_unit_test (NAME    StringUtils
                  PREFIX  ${dirName}
                  SOURCES StringUtils.cpp
                  SERIAL_ONLY)

# Check HessianVec fad
create_unit_test (NAME    HessianVecFad
                  PREFIX  ${dirName}
                  SOURCES HessianVecFad.cpp
                  SERIAL_ONLY)

#####################################################################

add_subdirectory(disc)
add_subdirectory(evaluators)
