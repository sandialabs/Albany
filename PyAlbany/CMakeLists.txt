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
                        "import sys; print(sys.version[:3])"
  OUTPUT_VARIABLE PYTHON_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

MESSAGE("  -- PYTHON_VERSION = ${PYTHON_VERSION}")

add_subdirectory( src )
add_subdirectory( swig )

# The CMAKE install logic of PyAlbany is based on the one of PyTrilinos.

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

# Python files to install
FILE(GLOB PyAlbanyPyFiles ${CMAKE_CURRENT_SOURCE_DIR}/python/*.py)

INSTALL(FILES
  ${PyAlbanyPyFiles}
  ${CMAKE_CURRENT_BINARY_DIR}/swig/wpyalbany.py
  ${CMAKE_CURRENT_BINARY_DIR}/swig/_wpyalbany.so
  DESTINATION ${PyAlbany_INSTALL_DIR})

MESSAGE("End of the configuration of PyAlbany")

IF (ALBANY_PYTHON_TESTS)
  SET(PYALBANY_PYTHONPATH "PYTHONPATH=${Trilinos_LIB_DIRS}/python${PYTHON_VERSION}/site-packages:${CMAKE_CURRENT_BINARY_DIR}/swig:$ENV{PYTHONPATH}")
  add_subdirectory( tests )
ENDIF()