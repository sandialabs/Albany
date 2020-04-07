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
  ${TRILINOS_SOURCE_DIR}/packages/phalanx/test/EvaluatorUnitTester
)

# Files in Albany to be built or are needed
SET(SOURCES
          ./EvaluatorUnitTester.cpp
          ${TRILINOS_SOURCE_DIR}/packages/phalanx/test/Utilities/Phalanx_UnitTestMain.cpp
)

SET(HEADERS
          ${CMAKE_SOURCE_DIR}/src/evaluators/utility/PHAL_ComputeBasisFunctions.hpp
)

# Files in PHX to be built as temporary examples
SET(PHX_SOURCES
)

SET(PHX_HEADERS
          ${TRILINOS_SOURCE_DIR}/packages/phalanx/test/EvaluatorUnitTester/SimpleEvaluator.hpp
          ${TRILINOS_SOURCE_DIR}/packages/phalanx/test/EvaluatorUnitTester/SimpleEvaluator_Def.hpp
          ${TRILINOS_SOURCE_DIR}/packages/phalanx/test/EvaluatorUnitTester/DuplicateFieldEvaluator.hpp
          ${TRILINOS_SOURCE_DIR}/packages/phalanx/test/EvaluatorUnitTester/DuplicateFieldEvaluator_Def.hpp
          ${TRILINOS_SOURCE_DIR}/packages/phalanx/test/EvaluatorUnitTester/MyTraits.hpp
)

LINK_DIRECTORIES (${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})

# Duplicate Roger's Phalanx EvaluatorUnitTester here for right now
ADD_EXECUTABLE(
  evaluator_unit_tester
  ${PHX_HEADERS} ${PHX_HEADERS}
  ${HEADERS} ${SOURCES}
)

set_target_properties(evaluator_unit_tester PROPERTIES
  PUBLIC_HEADER "${HEADERS} ${PHX_HEADERS}")

TARGET_LINK_LIBRARIES(evaluator_unit_tester albanyLib ${ALB_TRILINOS_LIBS} ${Trilinos_EXTRA_LD_FLAGS})

# We should always run the unit tests in both serial and parallel if possible (they should run quickly)
IF (ALBANY_MPI)
  ADD_TEST(
    Albany_Serial_Unit_Test ${SERIAL_CALL} ${CMAKE_CURRENT_BINARY_DIR}/evaluator_unit_tester
  )
  ADD_TEST(
    Albany_Parallel_Unit_Test ${PARALLEL_CALL} ${CMAKE_CURRENT_BINARY_DIR}/evaluator_unit_tester
  )
ELSE(ALBANY_MPI)
  ADD_TEST(
    Albany_Unit_Test ${CMAKE_CURRENT_BINARY_DIR}/evaluator_unit_tester
  )
ENDIF(ALBANY_MPI)

