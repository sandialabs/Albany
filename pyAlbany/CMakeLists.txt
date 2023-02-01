##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

MESSAGE("\nStart the configuration of PyAlbany")

OPTION(PYALBANY_USE_PYTHON3 "PyAlbany uses python 3" OFF)

MESSAGE("-- PYTHON_EXECUTABLE:")
IF(NOT DEFINED PYTHON_EXECUTABLE)
  IF(PYALBANY_USE_PYTHON3)
    find_program(PYTHON_EXECUTABLE
        NAMES python3 python
        )
  ELSE()
    find_program(PYTHON_EXECUTABLE
        NAMES python2 python
        )
  ENDIF()
  MESSAGE("  -- CMake has set: PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")
ELSE()
  MESSAGE("  -- User has set: PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")
ENDIF()

# Get the python version
EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} -c
                        "import sys; print(sys.version_info.major)"
  OUTPUT_VARIABLE PYTHON_MAJOR_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the python version
EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} -c
                        "import sys; print(sys.version_info.minor)"
  OUTPUT_VARIABLE PYTHON_MINOR_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

SET(PYTHON_VERSION ${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION})

MESSAGE("  -- PYTHON_VERSION = ${PYTHON_VERSION}")
MESSAGE("  -- PYTHON_MAJOR_VERSION = ${PYTHON_MAJOR_VERSION}")

SET(PYBIND11_PYTHON_VERSION ${PYTHON_VERSION})

# Determine the PyAlbany default installation directory prefix.  If
# CMAKE_INSTALL_PREFIX is set by the user, then use it.  Else the
# prefix is set to the value of python's sys.prefix.
EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print(sys.prefix)"
  OUTPUT_VARIABLE PYTHON_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(PyAlbany_DEFAULT_INSTALL_PREFIX ${PYTHON_PREFIX})
ELSE()
  SET(PyAlbany_DEFAULT_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
ENDIF()

# Set the PyAlbany install prefix
SET(PyAlbany_INSTALL_PREFIX ${PyAlbany_DEFAULT_INSTALL_PREFIX}
  CACHE PATH "The path prefix for where PyAlbany will be installed, e.g. /usr/local")

# Determine the install directory
SET(PyAlbany_INSTALL_DIR
  ${PyAlbany_INSTALL_PREFIX}/lib/python${PYTHON_VERSION}/site-packages/PyAlbany
  )
MESSAGE("-- PyAlbany installation path: ${PyAlbany_INSTALL_DIR}")

IF (NOT (ALBANY_LIBRARIES_ONLY OR DISABLE_ALBANY_TESTS) )
  set (ALBANY_PYTHON_TESTS TRUE)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/PyAlbany)
ENDIF()

MESSAGE("-- Check required dependencies:")

# Retrieve the Pybind11 version
#EXECUTE_PROCESS(COMMAND
#  ${PYTHON_EXECUTABLE} -c "import pybind11; print(pybind11.__version__)"
#  OUTPUT_VARIABLE PYBIND11_VERSION
#  ERROR_VARIABLE  PYBIND11_VERSION_ERROR
#  OUTPUT_STRIP_TRAILING_WHITESPACE
#  )

#IF(NOT PYBIND11_VERSION_ERROR)
#  MESSAGE("  -- Pybind11 Enabled.")
#ELSE()
#  MESSAGE(FATAL_ERROR "PYBIND11_VERSION_ERROR is defined; the python executable cannot access pybind11.")
#ENDIF()

# Retrieve the Mpi4Py version
EXECUTE_PROCESS(COMMAND
  ${PYTHON_EXECUTABLE} -c "import mpi4py; print(mpi4py.__version__)"
  OUTPUT_VARIABLE Mpi4Py_VERSION
  ERROR_VARIABLE  Mpi4Py_VERSION_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )

IF(NOT Mpi4Py_VERSION_ERROR)
  MESSAGE("  -- Mpi4Py Enabled.")
ELSE()
  MESSAGE(FATAL_ERROR "Mpi4Py_VERSION_ERROR is defined; the python executable cannot access mpi4py.")
ENDIF()

# Retrieve the Numpy version
EXECUTE_PROCESS(COMMAND
  ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.__version__)"
  OUTPUT_VARIABLE NUMPY_VERSION
  ERROR_VARIABLE  NUMPY_VERSION_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )

IF(NOT NUMPY_VERSION_ERROR)
  MESSAGE("  -- Numpy Enabled.")
ELSE()
  MESSAGE(FATAL_ERROR "NUMPY_VERSION_ERROR is defined; the python executable cannot access numpy.")
ENDIF()

MESSAGE("-- PyAlbany uses pybind11 ${PYBIND11_VERSION}, mpi4py ${Mpi4Py_VERSION}, and numpy ${NUMPY_VERSION}.")

MESSAGE("-- Check optional dependencies:")

EXECUTE_PROCESS(COMMAND
  PYTHONPATH=${Trilinos_LIB_DIRS} ${PYTHON_EXECUTABLE} -c "import exomerge${PYTHON_MAJOR_VERSION}; print(exomerge${PYTHON_MAJOR_VERSION}.__version__)"
  OUTPUT_VARIABLE Exomerge_VERSION
  ERROR_VARIABLE  Exomerge_VERSION_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )

IF(NOT Exomerge_VERSION_ERROR)
  MESSAGE("  -- Exomerge Enabled.")
  SET(PYALBANY_EXOMERGE_ENABLED true)
ELSE()
  MESSAGE("  -- Exomerge NOT Enabled.")
  SET(PYALBANY_EXOMERGE_ENABLED false)
ENDIF()

IF(ALBANY_PYTHON_TESTS)
  MESSAGE("-- ALBANY_PYTHON_TESTS Enabled.")
ELSE()
  MESSAGE("-- ALBANY_PYTHON_TESTS NOT Enabled.")
ENDIF()

add_subdirectory( src )

# Python files to install
FILE(GLOB PyAlbanyPyFiles ${CMAKE_CURRENT_SOURCE_DIR}/python/*.py)

IF(NOT PYALBANY_EXOMERGE_ENABLED)
  list(REMOVE_ITEM PyAlbanyPyFiles ${CMAKE_CURRENT_SOURCE_DIR}/python/FEM_postprocess.py)
ENDIF()


INSTALL(FILES
  ${PyAlbanyPyFiles}
  DESTINATION ${PyAlbany_INSTALL_DIR})

MESSAGE("End of the configuration of PyAlbany")

IF (ALBANY_PYTHON_TESTS)
  file(COPY ${PyAlbanyPyFiles} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/PyAlbany/.)
  SET(PYALBANY_PYTHONPATH "PYTHONPATH=${Trilinos_LIB_DIRS}/:${CMAKE_CURRENT_BINARY_DIR}:$ENV{PYTHONPATH}")
  add_subdirectory( tests )
ENDIF()
