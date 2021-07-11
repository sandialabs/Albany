#*****************************************************************//
#    Albany 3.0:  Copyright 2016 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//

# Copy Exodus test file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2D_Blk_Test.e
                 ${CMAKE_CURRENT_BINARY_DIR}/2D_Blk_Test.e COPYONLY)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.msh
                 ${CMAKE_CURRENT_BINARY_DIR}/cube.msh COPYONLY)

INCLUDE_DIRECTORIES(
  ${Trilinos_INCLUDE_DIRS}  
  ${Trilinos_TPL_INCLUDE_DIRS} 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/disc
  ${CMAKE_SOURCE_DIR}/src
)

# Files in Albany to be built or are needed
SET(SOURCES
          ./UnitTest_BlockedDOFManager.cpp
          ../Albany_UnitTestMain.cpp
)

SET(HEADERS
)

LINK_DIRECTORIES (${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})

ADD_EXECUTABLE(
  disc_unit_tester
  ${HEADERS} ${SOURCES}
)

set_target_properties(disc_unit_tester PROPERTIES
  PUBLIC_HEADER "${HEADERS}")

TARGET_LINK_LIBRARIES(disc_unit_tester albanyLib ${ALB_TRILINOS_LIBS} ${Trilinos_EXTRA_LD_FLAGS})

set(PYTHON_TEST TRUE)

# If not python executable is set, find the one in the path (if any):
IF(NOT DEFINED PYTHON_EXECUTABLE)
  find_program(PYTHON_EXECUTABLE
      NAMES python3 python
      )
  MESSAGE("  -- CMake has set: PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")
ENDIF()

# Get the python version
EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} -c
                        "import sys; print(sys.version[:3])"
  OUTPUT_VARIABLE PYTHON_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
IF(${PYTHON_VERSION} VERSION_LESS 2.7)
  set(PYTHON_TEST FALSE)
  MESSAGE("The PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE} is older than 2.7.0, the test BlockDiscretization_Jacobian_Unit_Test will NOT be run.")
ELSE()
  # Verify that numpy and scipy are available:
  EXECUTE_PROCESS(COMMAND
    ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.__version__)"
    OUTPUT_VARIABLE NUMPY_VERSION
    ERROR_VARIABLE  NUMPY_VERSION_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  EXECUTE_PROCESS(COMMAND
    ${PYTHON_EXECUTABLE} -c "import scipy; print(scipy.__version__)"
    OUTPUT_VARIABLE SCIPY_VERSION
    ERROR_VARIABLE  SCIPY_VERSION_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  IF(NUMPY_VERSION_ERROR OR SCIPY_VERSION_ERROR)
    set(PYTHON_TEST FALSE)
    MESSAGE("The PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE} does not support numpy or scipy, the test BlockDiscretization_Jacobian_Unit_Test will NOT be run.")
  ELSE()
    IF(${SCIPY_VERSION} VERSION_LESS 1.2.1 OR ${NUMPY_VERSION} VERSION_LESS 1.15.1)
      set(PYTHON_TEST FALSE)
      MESSAGE("The PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE} supports numpy and scipy but one of them is too old, the test BlockDiscretization_Jacobian_Unit_Test will NOT be run.")
    ELSE()
      MESSAGE("The PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE} supports numpy and scipy, the test BlockDiscretization_Jacobian_Unit_Test will be run.")
    ENDIF()
  ENDIF()
ENDIF()

IF(${PYTHON_TEST})
  SET(SOURCES
            ./UnitTest_BlockedJacobian.cpp
            ../Albany_UnitTestMain.cpp
  )

  ADD_EXECUTABLE(
    blockJacobian_unit_tester
    ${HEADERS} ${SOURCES}
  )

  set_target_properties(blockJacobian_unit_tester PROPERTIES
    PUBLIC_HEADER "${HEADERS}")

  TARGET_LINK_LIBRARIES(blockJacobian_unit_tester albanyLib ${ALB_TRILINOS_LIBS} ${Trilinos_EXTRA_LD_FLAGS})

  set(TESTFILES matrices_comparison.py)

  file(COPY ${TESTFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/.)
  ADD_TEST(BlockDiscretization_Jacobian_Unit_Test "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/matrices_comparison.py")
ENDIF()

# We should always run the unit tests in both serial and parallel if possible (they should run quickly)
IF (ALBANY_MPI)
  ADD_TEST(
    Discretization_Serial_Unit_Test ${SERIAL_CALL} ${CMAKE_CURRENT_BINARY_DIR}/disc_unit_tester
  )
# The parallel test does not run properly.
# It is set up to use IOSS to decompose the exodus file ("Use Serial Mesh = 1"), but
# mpirun -np 2 ./disc_unit_tester
# 
# Throws:
#
# p=0: *** Caught standard std::exception of type 'std::logic_error' :
# 
#  /ascldap/users/gahanse/Codes/Albany/src/utility/Albany_ThyraCrsMatrixFactory.cpp:78:
#  
#  Throw number = 1
#  
#  Throw test that evaluated to true: !isOneToOne(m_domain_vs)
#  
#  [ThyraCrsMatrixFactory] Error! When providing an overlapped domain vs, the domain vs must be one-to-one.
#
  ADD_TEST(
    Discretization_Parallel_Unit_Test ${PARALLEL_CALL} ${CMAKE_CURRENT_BINARY_DIR}/disc_unit_tester
  )
ELSE(ALBANY_MPI)
  ADD_TEST(
    Discretization_Unit_Test ${CMAKE_CURRENT_BINARY_DIR}/disc_unit_tester
  )
ENDIF(ALBANY_MPI)

