#*****************************************************************//
#    Albany 3.0:  Copyright 2016 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//

INCLUDE_DIRECTORIES(
  ${Trilinos_INCLUDE_DIRS}  
  ${Trilinos_TPL_INCLUDE_DIRS} 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/evaluators/utility
  ${CMAKE_SOURCE_DIR}/src/evaluators/interpolation
  ${CMAKE_SOURCE_DIR}/src/problems
  ${CMAKE_SOURCE_DIR}/src/disc
  ${CMAKE_SOURCE_DIR}/src/utility
  ${CMAKE_SOURCE_DIR}/src
)

# Files in Albany to be built or are needed
SET(SOURCES
          ./EvaluatorUnitTester.cpp
          ../Albany_UnitTestMain.cpp
          ${CMAKE_SOURCE_DIR}/src/evaluators/interpolation/PHAL_DOFInterpolation.cpp
)

SET(SOURCES_gatherSolution
          ./gatherSolution.cpp
          ../Albany_UnitTestMain.cpp
          ${CMAKE_SOURCE_DIR}/src/evaluators/interpolation/PHAL_DOFInterpolation.cpp
)

SET(SOURCES_gatherDistributedParameters
          ./gatherDistributedParameters.cpp
          ../Albany_UnitTestMain.cpp
          ${CMAKE_SOURCE_DIR}/src/evaluators/interpolation/PHAL_DOFInterpolation.cpp
)

SET(HEADERS
          ${CMAKE_SOURCE_DIR}/src/evaluators/utility/PHAL_ComputeBasisFunctions.hpp
          ${CMAKE_SOURCE_DIR}/src/evaluators/interpolation/PHAL_DOFInterpolation.hpp
          ${CMAKE_SOURCE_DIR}/src/evaluators/interpolation/PHAL_DOFInterpolation_Def.hpp
)

LINK_DIRECTORIES (${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})

# Duplicate Roger's Phalanx EvaluatorUnitTester here for right now
ADD_EXECUTABLE(
  evaluator_unit_tester
  ${HEADERS} ${SOURCES}
)

ADD_EXECUTABLE(
  gatherSolution_unit_tester
  ${HEADERS} ${SOURCES_gatherSolution}
)

ADD_EXECUTABLE(
  gatherDistributedParameters_unit_tester
  ${HEADERS} ${SOURCES_gatherDistributedParameters}
)

set_target_properties(evaluator_unit_tester PROPERTIES
  PUBLIC_HEADER "${HEADERS}")

TARGET_LINK_LIBRARIES(evaluator_unit_tester albanyLib ${ALB_TRILINOS_LIBS} ${Trilinos_EXTRA_LD_FLAGS})

set_target_properties(gatherSolution_unit_tester PROPERTIES
  PUBLIC_HEADER "${HEADERS}")

TARGET_LINK_LIBRARIES(gatherSolution_unit_tester albanyLib ${ALB_TRILINOS_LIBS} ${Trilinos_EXTRA_LD_FLAGS})

set_target_properties(gatherDistributedParameters_unit_tester PROPERTIES
  PUBLIC_HEADER "${HEADERS}")

TARGET_LINK_LIBRARIES(gatherDistributedParameters_unit_tester albanyLib ${ALB_TRILINOS_LIBS} ${Trilinos_EXTRA_LD_FLAGS})

# We should always run the unit tests in both serial and parallel if possible (they should run quickly)
IF (ALBANY_MPI)
  ADD_TEST(
    Albany_Serial_Unit_Test ${SERIAL_CALL} ${CMAKE_CURRENT_BINARY_DIR}/evaluator_unit_tester
  )
  ADD_TEST(
    Albany_Parallel_Unit_Test ${PARALLEL_CALL} ${CMAKE_CURRENT_BINARY_DIR}/evaluator_unit_tester
  )
  ADD_TEST(
    Albany_Serial_GatherSolution_Unit_Test ${SERIAL_CALL} ${CMAKE_CURRENT_BINARY_DIR}/gatherSolution_unit_tester
  )
  ADD_TEST(
    Albany_Parallel_GatherSolution_Unit_Test ${PARALLEL_CALL} ${CMAKE_CURRENT_BINARY_DIR}/gatherSolution_unit_tester
  )
  ADD_TEST(
    Albany_Serial_GatherDistributedParameters_Unit_Test ${SERIAL_CALL} ${CMAKE_CURRENT_BINARY_DIR}/gatherDistributedParameters_unit_tester
  )
  ADD_TEST(
    Albany_Parallel_GatherDistributedParameters_Unit_Test ${PARALLEL_CALL} ${CMAKE_CURRENT_BINARY_DIR}/gatherDistributedParameters_unit_tester
  )
ELSE(ALBANY_MPI)
  ADD_TEST(
    Albany_Unit_Test ${CMAKE_CURRENT_BINARY_DIR}/evaluator_unit_tester
  )
  ADD_TEST(
    Albany_GatherSolution_Unit_Test ${CMAKE_CURRENT_BINARY_DIR}/gatherSolution_unit_tester
  )
  ADD_TEST(
    Albany_GatherDistributedParameters_Unit_Test ${CMAKE_CURRENT_BINARY_DIR}/gatherDistributedParameters_unit_tester
  )
ENDIF(ALBANY_MPI)

