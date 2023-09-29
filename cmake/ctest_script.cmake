cmake_minimum_required(VERSION 3.9)

#########################################
#     CTest configuration variables     #
#########################################

# Set some defaults for CTest behaviors.
# Note: users can run ctest locally passing -DVAR=value

if (NOT DEFINED CTEST_DO_SUBMIT)
  set (CTEST_DO_SUBMIT OFF)
endif()

if (NOT DEFINED CTEST_TEST_TYPE)
  set (CTEST_TEST_TYPE Nightly)
endif()

if (NOT DEFINED CTEST_TEST_TIMEOUT)
  set (CTEST_TEST_TIMEOUT 600)
endif()

if (NOT DEFINED CTEST_CMAKE_GENERATOR)
  set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
endif()

if (NOT DEFINED CTEST_PARALLEL_LEVEL)
  set (CTEST_PARALLEL_LEVEL 8)
endif()

get_filename_component(Albany_ROOT ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)
set(CTEST_SOURCE_DIRECTORY "${Albany_ROOT}")
if (NOT DEFINED CTEST_BINARY_DIR)
  set(CTEST_BINARY_DIRECTORY ${Albany_ROOT}/build)
endif()

# Over-write default limit for output posted to CDash site
set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE 5000000)
set(CTEST_CUSTOM_MAXIMUM_FAILED_TEST_OUTPUT_SIZE 5000000)

if (NOT DEFINED CTEST_MACHINE)
  # If user did not pass a machine, we will not submit
  set (CTEST_MACHINE "Generic")
  set (CTEST_DO_SUBMIT OFF)
endif()

#########################################
#    Albany configuration variables     #
#########################################

# Set some defaults for the build type. This can be overwritten by
# running ctest with -DVAR=value
if (NOT DEFINED ENABLE_DEMO_PDES)
  set (ENABLE_DEMO_PDES ON)
endif()
if (NOT DEFINED ENABLE_LANDICE)
  set (ENABLE_LANDICE ON)
endif()
if (NOT DEFINED ENABLE_MPAS_INTERFACE)
  set (ENABLE_MPAS_INTERFACE ON)
endif()
if (NOT DEFINED ENABLE_MESH_DEPENDS_ON_PARAMETERS)
  set (ENABLE_MESH_DEPENDS_ON_PARAMETERS OFF)
endif()
if (NOT DEFINED MAX_NUMPROCS)
  set (MAX_NUMPROCS 8)
endif()

# The following are mandatory, so error out if they're not set
if (NOT DEFINED CMAKE_CXX_COMPILER)
  message (FATAL_ERROR "You did not pass -DCMAKE_CXX_COMPILER=<compiler>")
endif()
if (NOT DEFINED CMAKE_C_COMPILER)
  message (FATAL_ERROR "You did not pass -DCMAKE_C_COMPILER=<compiler>")
endif()
if (NOT DEFINED ALBANY_TRILINOS_DIR)
  message (FATAL_ERROR "You did not pass -DALBANY_TRILINOS_DIR=<path>")
endif()

#########################################
#      Step 0: start ctest session      #
#########################################

ctest_start(${CTEST_TEST_TYPE} GROUP ${CTEST_MACHINE})

#########################################
#          Step 1: configure            #
#########################################

string (CONCAT MSG
  "Running configure step with:\n"
  " - CMAKE_CXX_COMPILER                : ${CMAKE_CXX_COMPILER}\n"
  " - CMAKE_C_COMPILER                  : ${CMAKE_C_COMPILER}\n"
  " - ALBANY_TRILINOS_DIR               : ${ALBANY_TRILINOS_DIR}\n"
  " - ALBANY_MPI_EXEC_MAX_NUMPROCS      : ${MAX_NUMPROCS}\n"
  " - ENABLE_DEMO_PDES                  : ${ENABLE_DEMO_PDES}\n"
  " - ENABLE_LANDICE                    : ${ENABLE_LANDICE}\n"
  " - ENABLE_MPAS_INTERFACE             : ${ENABLE_MPAS_INTERFACE}\n"
  " - ENABLE_MESH_DEPENDS_ON_PARAMETERS : ${ENABLE_MESH_DEPENDS_ON_PARAMETERS}\n")
message (${MSG})
set (CONFIGURE_OPTIONS
  -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
  -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
  -DALBANY_TRILINOS_DIR:PATH=${ALBANY_TRILINOS_DIR}
  -DALBANY_MPI_EXEC_MAX_NUMPROCS:STRING=${MAX_NUMPROCS}
  -DENABLE_DEMO_PDES:BOOL=${ENABLE_DEMO_PDES}
  -DENABLE_LANDICE:BOOL=${ENABLE_LANDICE}
  -DENABLE_MPAS_INTERFACE:BOOL=${ENABLE_MPAS_INTERFACE}
  -DENABLE_MESH_DEPENDS_ON_PARAMETERS:BOOL=${ENABLE_MESH_DEPENDS_ON_PARAMETERS}
  )

ctest_configure(
  SOURCE ${CTEST_SOURCE_DIRECTORY}
  BUILD  ${CTEST_BINARY_DIRECTORY}
  OPTIONS "${CONFIGURE_OPTIONS}"
  RETURN_VALUE CONFIG_ERROR
  APPEND
)

if (CTEST_DO_SUBMIT)
  ctest_submit (
    PARTS        Configure
    RETURN_VALUE SUBMIT_ERROR
  )
  if (SUBMIT_ERROR)
    message(FATAL_ERROR "Cannot submit Albany configure results!")
  endif ()
endif ()

if (CONFIG_ERROR)
  message(FATAL_ERROR "Cannot configure Albany build!")
endif ()

#########################################
#        Step 2: build/install          #
#########################################

set (CTEST_BUILD_TARGET install)
ctest_build(
  BUILD           ${CTEST_BINARY_DIRECTORY}
  RETURN_VALUE    BUILD_ERROR
  NUMBER_ERRORS   BUILD_NUM_ERRORS
  NUMBER_WARNINGS BUILD_NUM_WARNINGS
  PARALLEL_LEVEL  ${CTEST_PARALLEL_LEVEL}
  APPEND
)

if (CTEST_DO_SUBMIT)
  ctest_submit (
    PARTS        Build
    RETURN_VALUE SUBMIT_ERROR
  )

  if (SUBMIT_ERROR)
    message(FATAL_ERROR "Cannot submit Albany build results!")
  endif ()
endif ()

message("Build step completed")
message("  - num warnings: ${BUILD_NUM_WARNINGS}")
message("  - num errors  : ${BUILD_NUM_ERRORS}")

if (BUILD_ERROR)
  message(FATAL_ERROR "Cannot build Albany!")
endif ()

#########################################
#          Step 3: run tests            #
#########################################

ctest_test(
  BUILD          ${CTEST_BINARY_DIRECTORY}
  PARALLEL_LEVEL ${CTEST_PARALLEL_LEVEL}
  RETURN_VALUE   TEST_ERROR
  APPEND
)

if (CTEST_DO_SUBMIT)
  ctest_submit (
    PARTS        Test
    RETURN_VALUE SUBMIT_ERROR
  )

  if (SUBMIT_ERROR)
    message(FATAL_ERROR "Cannot submit Albany test results!")
  endif ()
endif ()

message ("Albany testing completed successfully")
