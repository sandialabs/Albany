##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

#####################################################################
# NullSpaceUtils unit tests
INCLUDE_DIRECTORIES(
  ${Trilinos_INCLUDE_DIRS}
  ${Trilinos_TPL_INCLUDE_DIRS}
  ${CMAKE_BINARY_DIR}/src
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}
)

SET(SOURCES
  ./UnitTest_NullSpaceUtils.cpp
  ./Albany_UnitTestMain.cpp
)

#LINK_DIRECTORIES(${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})

ADD_EXECUTABLE(
  nullSpaceUtils_unit_tester
  ${SOURCES}
)

TARGET_LINK_LIBRARIES(nullSpaceUtils_unit_tester albanyLib ${ALB_TRILINOS_LIBS} ${Trilinos_EXTRA_LD_FLAGS})

ADD_TEST(
  NullSpaceUtils_Serial_Unit_Test ${SERIAL_CALL} ${CMAKE_CURRENT_BINARY_DIR}/nullSpaceUtils_unit_tester
)
IF(ALBANY_MPI)
  ADD_TEST(
    NullSpaceUtils_Parallel_Unit_Test ${PARALLEL_CALL} ${CMAKE_CURRENT_BINARY_DIR}/nullSpaceUtils_unit_tester
  )
ENDIF(ALBANY_MPI)

#####################################################################

add_subdirectory(evaluators)
add_subdirectory(disc)

