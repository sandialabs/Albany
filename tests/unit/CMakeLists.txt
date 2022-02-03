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

# NullSpaceUtils unit tests
add_executable (null_space_utils ./UnitTest_NullSpaceUtils.cpp)
target_link_libraries (null_space_utils PUBLIC albany_ut_main)
add_test(NAME NullSpaceUtils_Serial_Unit_Test
         COMMAND ${SERIAL_CALL} null_space_utils)
if(ALBANY_MPI)
  add_test(NAME NullSpaceUtils_Parallel_Unit_Test
    COMMAND ${PARALLEL_CALL} null_space_utils)
endif(ALBANY_MPI)

#####################################################################

add_subdirectory(evaluators)
add_subdirectory(disc)

