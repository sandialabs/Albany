##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

#######################################
#          Unit tests main            #
#######################################

# We build the main as a library, then link it to each executable
add_library (albany_ut_main ./Albany_UnitTestMain.cpp)
target_link_libraries (albany_ut_main PUBLIC albanyLib)

#######################################
#       Individual unit tests         #
#######################################


get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${dirName})

# NullSpaceUtils unit tests
add_executable (null_space_utils ./UnitTest_NullSpaceUtils.cpp)
target_link_libraries (null_space_utils PUBLIC albany_ut_main)
add_test(NAME ${testNameRoot}_NullSpaceUtils_Serial
         COMMAND ${SERIAL_CALL} null_space_utils)

if(ALBANY_MPI)
	add_test(NAME ${testNameRoot}_NullSpaceUtils_Parallel
    COMMAND ${PARALLEL_CALL} null_space_utils)
endif(ALBANY_MPI)

# StringUtils unit tests (no MPI test, since it's pointless)
add_executable (string_utils ./UnitTest_StringUtils.cpp)
target_link_libraries (string_utils PUBLIC albany_ut_main)
add_test(NAME ${testNameRoot}_StringUtils_Serial
         COMMAND ${SERIAL_CALL} string_utils)

#####################################################################

add_subdirectory(evaluators)
add_subdirectory(disc)

