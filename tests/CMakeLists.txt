##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

MESSAGE ("\n ***********************")
MESSAGE (" *** Testing Options ***")
MESSAGE (" ***********************\n")

# Add option to enable Albany unit tests.
OPTION(ENABLE_UNIT_TESTS "Flag to turn on Unit Tests" ON)
IF (ENABLE_UNIT_TESTS)
  MESSAGE("-- Unit Tests                Enabled, adding tests/*/UnitTests")
  SET(ALBANY_UNIT_TESTS TRUE)
ELSE()
  MESSAGE("-- Unit Tests            NOT Enabled.")
  SET(ALBANY_UNIT_TESTS FALSE)
ENDIF()

# Add option to enable more expensive tests.
OPTION(ENABLE_LARGE_TESTS "Run more time-consuming tests" OFF)
IF (ENABLE_LARGE_TESTS)
  set (ALBANY_LARGE_TESTS ON)
  MESSAGE("-- Large Tests               Enabled/")
ELSE()
  MESSAGE("-- Large Tests           NOT Enabled.")
ENDIF()

#  Timeout real long running tests
IF(ALBANY_CTEST_TIMEOUT)
    MESSAGE("-- Setting maximum test run time to be       ${ALBANY_CTEST_TIMEOUT} seconds")
ELSE()
    MESSAGE("-- No limits are set on maximum test run time")
ENDIF()

OPTION(ALBANY_PARALLEL_ONLY "Only run the parallel tests" OFF)

#Add option to run any pretest executables in a specific environment (mpiexec.hydra, srun, or other)
IF (DEFINED ALBANY_PRETEST_EXEC)
  MESSAGE("-- ALBANY_PRETEST_EXEC is ${ALBANY_PRETEST_EXEC}")
ELSE()
  # Not set in configure - set it to be empty
  SET(ALBANY_PRETEST_EXEC "" CACHE STRING "Command to use to execute pretests")
ENDIF()

#Add option to specify path to desired SEACAS utils
IF (DEFINED ALBANY_SEACAS_PATH)
  MESSAGE("-- ALBANY_SEACAS_PATH is ${ALBANY_SEACAS_PATH}")
ENDIF()


# Setup paths for SEACAS tools needed to run some of the tests
IF (ALBANY_SEACAS)
  IF(DEFINED ALBANY_SEACAS_PATH)
    find_program(
         SEACAS_EPU
         NAMES epu
         PATHS ${ALBANY_SEACAS_PATH}
    )
    find_program(
         SEACAS_EXODIFF
         NAMES exodiff
         PATHS ${ALBANY_SEACAS_PATH}
    )
    find_program(
         SEACAS_ALGEBRA
         NAMES algebra
         PATHS ${ALBANY_SEACAS_PATH}
    )
    find_program(
         SEACAS_DECOMP
         NAMES decomp
         PATHS ${ALBANY_SEACAS_PATH}
    )
  ELSE()
    find_program(
         SEACAS_EPU
         NAMES epu
         PATHS ${Trilinos_BIN_DIRS} ENV PATH
    )
    find_program(
         SEACAS_EXODIFF
         NAMES exodiff
         PATHS ${Trilinos_BIN_DIRS} ENV PATH
    )
    find_program(
         SEACAS_ALGEBRA
         NAMES algebra
         PATHS ${Trilinos_BIN_DIRS} ENV PATH
    )
    find_program(
         SEACAS_DECOMP
         NAMES decomp
         PATHS ${Trilinos_BIN_DIRS} ENV PATH
    )
  ENDIF()
  MESSAGE("-- SEACAS_ALGEBRA = ${SEACAS_ALGEBRA}")
  MESSAGE("-- SEACAS_DECOMP  = ${SEACAS_DECOMP}")
  MESSAGE("-- SEACAS_EPU     = ${SEACAS_EPU}")
  MESSAGE("-- SEACAS_EXODIFF = ${SEACAS_EXODIFF}")
ENDIF()


# Paths to the actual executables
set(AlbanyPath                         ${Albany_BINARY_DIR}/src/Albany)
set(AlbanyAnalysisPath                 ${Albany_BINARY_DIR}/src/AlbanyAnalysis)

set(SerialAlbany.exe                 ${SERIAL_CALL} ${AlbanyPath})
set(SerialAlbanyAnalysis.exe         ${SERIAL_CALL} ${AlbanyAnalysisPath})
# Do not test on greater than Trilinos_MPI_EXEC_MAX_NUMPROCS configured in Trilinos build
# or explicitly given at Albany configure time -D ALBANY_MPI_EXEC_MAX_NUMPROCS
IF(DEFINED MPIMNP)
  IF(MPIMNP LESS 8)
    SET(MAX_MPI_RANKS ${MPIMNP})
  ELSE(MPIMNP LESS 8)
    SET(MAX_MPI_RANKS 8)
  ENDIF(MPIMNP LESS 8)
ELSE(DEFINED MPIMNP)
    SET(MAX_MPI_RANKS 4)
ENDIF(DEFINED MPIMNP)
IF (ALBANY_ENABLE_CUDA) 
  set(Albany.exe                     ${PARALLEL_CALL} ${AlbanyPath} ${KOKKOS_NDEVICES})
  set(Albany8.exe                    ${MPIEX} ${MPIPRE} ${MPINPF} ${MAX_MPI_RANKS} ${MPIPOST} ${AlbanyPath} ${KOKKOS_NDEVICES})
  set(AlbanyAnalysis.exe             ${PARALLEL_CALL} ${AlbanyAnalysisPath} ${KOKKOS_NDEVICES})
ELSE() 
  set(Albany.exe                     ${PARALLEL_CALL} ${AlbanyPath})
  set(Albany8.exe                    ${MPIEX} ${MPIPRE} ${MPINPF} ${MAX_MPI_RANKS} ${MPIPOST} ${AlbanyPath})
  set(AlbanyAnalysis.exe             ${PARALLEL_CALL} ${AlbanyAnalysisPath})
ENDIF() 

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
  set(PYTHON_TEST_HELPERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python)
ENDIF()

add_subdirectory(corePDEs)

IF (ALBANY_UNIT_TESTS)
  add_subdirectory(unit)
ENDIF()

IF(ENABLE_DEMO_PDES)
  add_subdirectory(demoPDEs)
ENDIF()

IF(ENABLE_LANDICE AND ALBANY_SEACAS )
  add_subdirectory(landIce)
ENDIF()
