##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# -- Search for SWIG
FIND_PACKAGE(SWIG REQUIRED)

SET(CMAKE_SWIG_OUTDIR "${CMAKE_CURRENT_BINARY_DIR}")
INCLUDE(${SWIG_USE_FILE})

FILE(GLOB SRCS *.h *.cpp *.inl *.swg)
FILE(GLOB ISRCS *.i)

SET(CMAKE_SWIG_FLAGS "")
IF(NOT ${PYTHON_VERSION} VERSION_LESS 3.0)
  LIST(APPEND CMAKE_SWIG_FLAGS -py3)
ENDIF()
SET_SOURCE_FILES_PROPERTIES(${ISRCS} PROPERTIES CPLUSPLUS ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${TRILINOS_SOURCE_DIR}/packages/PyTrilinos/cmake)

FIND_PACKAGE(PythonInterp 2.5 REQUIRED)

MESSAGE("-- PYTHON_INCLUDE_PATH:")
IF(NOT DEFINED PYTHON_INCLUDE_PATH)
    FIND_PACKAGE(PythonInclude REQUIRED)
    MESSAGE("  -- CMake has set: PYTHON_INCLUDE_PATH = ${PYTHON_INCLUDE_PATH}")
    MESSAGE("  -- Please verify that ${PYTHON_INCLUDE_PATH} includes the Python.h file.")
    MESSAGE("  -- If not, specify the path manually setting PYTHON_INCLUDE_PATH to directory which includes the Python.h file.")
ELSE()
    MESSAGE("  -- User has set: PYTHON_INCLUDE_PATH = ${PYTHON_INCLUDE_PATH}")
ENDIF()
FIND_PACKAGE(NumPy 1.0 REQUIRED)
FIND_PACKAGE(Mpi4Py REQUIRED)

INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})

SET(SWINCFLAGS 
-I${Albany_SOURCE_DIR}/PyAlbany/src
-I${Albany_SOURCE_DIR}/src
-I${Albany_SOURCE_DIR}
-I${Trilinos_INCLUDE_DIRS}
)
SET_SOURCE_FILES_PROPERTIES(${ISRCS} PROPERTIES SWIG_FLAGS "${SWINCFLAGS}")

INCLUDE_DIRECTORIES( ${Albany_SOURCE_DIR}/PyAlbany/src
                     ${Albany_SOURCE_DIR}/src
                     ${Albany_SOURCE_DIR}/src/disc
                     ${Albany_SOURCE_DIR}/src/utility
                     ${Albany_SOURCE_DIR}/src/problems
                     ${Albany_SOURCE_DIR}/src/evaluators/bc
                     ${CMAKE_BINARY_DIR}/src
                     ${Trilinos_INCLUDE_DIRS}
                     ${Trilinos_TPL_INCLUDE_DIRS}
                     ${Mpi4Py_INCLUDE_DIR}
                     ${NUMPY_INCLUDE_DIR}
                     ${TRILINOS_SOURCE_DIR}/packages/PyTrilinos/src
                     ${TRILINOS_SOURCE_DIR}/packages/tpetra/classic/NodeAPI
                     ${TRILINOS_BUILD_DIR}/packages/PyTrilinos/doc/Doxygen)

SWIG_ADD_LIBRARY(wpyalbany LANGUAGE python SOURCES ${ISRCS} ${SRCS})

TARGET_INCLUDE_DIRECTORIES(_wpyalbany PRIVATE ${PYTHON_INCLUDE_PATH} ${NumPy_INCLUDE_DIR} ${Mpi4Py_INCLUDE_DIR})

SWIG_LINK_LIBRARIES(wpyalbany ${PYTHON_LIBRARIES} albanyPyInterface)
