
#cmake_minimum_required (VERSION 2.8)
set (CTEST_DO_SUBMIT ON)
set (CTEST_TEST_TYPE Nightly)
SET(CTEST_BUILD_OPTION "$ENV{BUILD_OPTION}")

# What to build and test
set (DOWNLOAD_ALBANY FALSE) 

if (1)
  # What to build and test
  IF(CTEST_BUILD_OPTION MATCHES "sfad4")
    set (BUILD_ALBANY_SFAD4 TRUE)
    set (BUILD_ALBANY_SFAD6 FALSE)
    set (BUILD_ALBANY_SFAD8 FALSE)
    set (BUILD_ALBANY_SFAD12 FALSE)
    set (CTEST_BUILD_NAME "waterman-CUDA-Albany-SFAD4")
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "sfad6")
    set (BUILD_ALBANY_SFAD4 FALSE)
    set (BUILD_ALBANY_SFAD6 TRUE)
    set (BUILD_ALBANY_SFAD8 FALSE)
    set (BUILD_ALBANY_SFAD12 FALSE)
    set (CTEST_BUILD_NAME "waterman-CUDA-Albany-SFAD6")
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "sfad8")
    set (BUILD_ALBANY_SFAD4 FALSE)
    set (BUILD_ALBANY_SFAD6 FALSE)
    set (BUILD_ALBANY_SFAD8 TRUE)
    set (BUILD_ALBANY_SFAD12 FALSE)
    set (CTEST_BUILD_NAME "waterman-CUDA-Albany-SFAD8")
  ENDIF()
 IF(CTEST_BUILD_OPTION MATCHES "sfad12")
    set (BUILD_ALBANY_SFAD4 FALSE)
    set (BUILD_ALBANY_SFAD6 FALSE)
    set (BUILD_ALBANY_SFAD8 FALSE)
    set (BUILD_ALBANY_SFAD12 TRUE)
    set (CTEST_BUILD_NAME "waterman-CUDA-Albany-SFAD12")
  ENDIF()
ENDIF()


# Begin User inputs:
set (CTEST_SITE "waterman.sandia.gov" ) # generally the output of hostname
set (CTEST_DASHBOARD_ROOT "$ENV{TEST_DIRECTORY}" ) # writable path
set (CTEST_SCRIPT_DIRECTORY "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set (CTEST_CMAKE_GENERATOR "Unix Makefiles" ) # What is your compilation apps ?
set (CTEST_CONFIGURATION  Release) # What type of build do you want ?

set (INITIAL_LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})

set (CTEST_PROJECT_NAME "Albany" )
set (CTEST_SOURCE_NAME repos)
set (CTEST_BINARY_NAME build)

set (CTEST_SOURCE_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_SOURCE_NAME}")
set (CTEST_BINARY_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_BINARY_NAME}")

if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}")
  file (MAKE_DIRECTORY "${CTEST_SOURCE_DIRECTORY}")
endif ()
if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}")
  file (MAKE_DIRECTORY "${CTEST_BINARY_DIRECTORY}")
endif ()

configure_file (${CTEST_SCRIPT_DIRECTORY}/CTestConfig.cmake
  ${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake COPYONLY)

set (CTEST_NIGHTLY_START_TIME "01:00:00 UTC")
set (CTEST_CMAKE_COMMAND "cmake")
set (CTEST_COMMAND "ctest -D ${CTEST_TEST_TYPE}")
set (CTEST_FLAGS "-j32")
SET (CTEST_BUILD_FLAGS "-j32")

set (CTEST_DROP_METHOD "https")

if (CTEST_DROP_METHOD STREQUAL "https")
  set (CTEST_DROP_SITE "sems-cdash-son.sandia.gov")
  set (CTEST_PROJECT_NAME "Albany")
  set (CTEST_DROP_LOCATION "/cdash/submit.php?project=Albany")
  set (CTEST_TRIGGER_SITE "")
  set (CTEST_DROP_SITE_CDASH TRUE)
endif ()

find_program (CTEST_GIT_COMMAND NAMES git)

set (Albany_REPOSITORY_LOCATION git@github.com:SNLComputation/Albany.git)
set (Trilinos_REPOSITORY_LOCATION git@github.com:trilinos/Trilinos.git)

#set (NVCC_WRAPPER "$ENV{jenkins_trilinos_dir}/packages/kokkos/config/nvcc_wrapper")
set (NVCC_WRAPPER ${CTEST_SCRIPT_DIRECTORY}/nvcc_wrapper_volta)
set (CUDA_MANAGED_FORCE_DEVICE_ALLOC 1)
set( CUDA_LAUNCH_BLOCKING 1)

if (CLEAN_BUILD)
  # Initial cache info
  set (CACHE_CONTENTS "
  SITE:STRING=${CTEST_SITE}
  CMAKE_TYPE:STRING=Release
  CMAKE_GENERATOR:INTERNAL=${CTEST_CMAKE_GENERATOR}
  TESTING:BOOL=OFF
  PRODUCT_REPO:STRING=${Albany_REPOSITORY_LOCATION}
  " )

  ctest_empty_binary_directory( "${CTEST_BINARY_DIRECTORY}" )
  file(WRITE "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt" "${CACHE_CONTENTS}")
endif ()


if (DOWNLOAD_ALBANY)

  set (CTEST_CHECKOUT_COMMAND)
  set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
  
  #
  # Get Albany
  #

  if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/Albany")
    execute_process (COMMAND "${CTEST_GIT_COMMAND}" 
      clone ${Albany_REPOSITORY_LOCATION} -b master ${CTEST_SOURCE_DIRECTORY}/Albany
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
    
    message(STATUS "out: ${_out}")
    message(STATUS "err: ${_err}")
    message(STATUS "res: ${HAD_ERROR}")
    if (HAD_ERROR)
      message(FATAL_ERROR "Cannot clone Albany repository!")
    endif ()
  endif ()

  set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")


endif ()


ctest_start(${CTEST_TEST_TYPE})

#
# Send the project structure to CDash
#

#if (CTEST_DO_SUBMIT)
#  ctest_submit (FILES "${CTEST_SCRIPT_DIRECTORY}/Project.xml"
#    RETURN_VALUE  HAD_ERROR
#    )

#  if (HAD_ERROR)
#    message(FATAL_ERROR "Cannot submit Albany Project.xml!")
#  endif ()
#endif ()

# 
# Set the common Trilinos config options & build Trilinos
# 

if (BUILD_ALBANY_SFAD4)

  # Configure the Albany build 
  #

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:FILEPATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF"
    "-DENABLE_DEMO_PDES:BOOL=ON"
    "-DENABLE_LANDICE:BOOL=ON"
    "-DENABLE_ALBANY_EPETRA:BOOL=OFF"
    "-DENABLE_PERFORMANCE_TESTS:BOOL=OFF"
    "-DALBANY_LIBRARIES_ONLY=OFF"
    "-DENABLE_KOKKOS_UNDER_DEVELOPMENT:BOOL=ON"
    "-DENABLE_FAD_TYPE:STRING='SFad'"
    "-DALBANY_SFAD_SIZE=4"
    "-DDISABLE_ALBANY_TESTS:BOOL=ON"
    )
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad4")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/AlbBuildSFad4)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad4"
    SOURCE "/home/projects/albany/waterman/repos/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    )
ENDIF()

if (BUILD_ALBANY_SFAD6)

  # Configure the Albany build 
  #

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:FILEPATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF"
    "-DENABLE_DEMO_PDES:BOOL=ON"
    "-DENABLE_LANDICE:BOOL=ON"
    "-DENABLE_ALBANY_EPETRA:BOOL=OFF"
    "-DENABLE_PERFORMANCE_TESTS:BOOL=OFF"
    "-DALBANY_LIBRARIES_ONLY=OFF"
    "-DENABLE_KOKKOS_UNDER_DEVELOPMENT:BOOL=ON"
    "-DENABLE_FAD_TYPE:STRING='SFad'"
    "-DALBANY_SFAD_SIZE=6"
    "-DDISABLE_ALBANY_TESTS:BOOL=ON"
    )
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad6")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/AlbBuildSFad6)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad6"
    SOURCE "/home/projects/albany/waterman/repos/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    )
ENDIF()

if (BUILD_ALBANY_SFAD8)

  # Configure the Albany build 
  #

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:FILEPATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF"
    "-DENABLE_DEMO_PDES:BOOL=ON"
    "-DENABLE_LANDICE:BOOL=ON"
    "-DENABLE_ALBANY_EPETRA:BOOL=OFF"
    "-DENABLE_PERFORMANCE_TESTS:BOOL=OFF"
    "-DALBANY_LIBRARIES_ONLY=OFF"
    "-DENABLE_KOKKOS_UNDER_DEVELOPMENT:BOOL=ON"
    "-DENABLE_FAD_TYPE:STRING='SFad'"
    "-DALBANY_SFAD_SIZE=8"
    "-DDISABLE_ALBANY_TESTS:BOOL=ON"
    )
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/AlbBuildSFad)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad"
    SOURCE "/home/projects/albany/waterman/repos/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    )
ENDIF()

if (BUILD_ALBANY_SFAD12)
  # Configure the Albany build 
  #

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:FILEPATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF"
    "-DENABLE_DEMO_PDES:BOOL=ON"
    "-DENABLE_LANDICE:BOOL=ON"
    "-DENABLE_ALBANY_EPETRA:BOOL=OFF"
    "-DENABLE_PERFORMANCE_TESTS:BOOL=OFF"
    "-DALBANY_LIBRARIES_ONLY=OFF"
    "-DENABLE_KOKKOS_UNDER_DEVELOPMENT:BOOL=ON"
    "-DENABLE_FAD_TYPE:STRING='SFad'"
    "-DALBANY_SFAD_SIZE=12"
    "-DDISABLE_ALBANY_TESTS:BOOL=ON"
    )
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad12")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/AlbBuildSFad12)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad12"
    SOURCE "/home/projects/albany/waterman/repos/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    )
ENDIF()

if (CTEST_DO_SUBMIT)
  ctest_submit (PARTS Configure
    RETURN_VALUE  S_HAD_ERROR
    )

  if (S_HAD_ERROR)
    message ("Cannot submit Albany configure results!")
  endif ()
endif ()


if (CTEST_DO_SUBMIT)
  ctest_submit (PARTS Configure
    RETURN_VALUE  S_HAD_ERROR
    )

  if (S_HAD_ERROR)
    message ("Cannot submit Albany configure results!")
  endif ()
endif ()

#
# Build the rest of Albany and install everything
#

set (CTEST_BUILD_TARGET all)
#set (CTEST_BUILD_TARGET install)

MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")


IF (BUILD_ALBANY_SFAD4)
  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad4"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )
ENDIF()
IF (BUILD_ALBANY_SFAD6)
  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad6"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )
ENDIF()
IF (BUILD_ALBANY_SFAD8)
  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )
ENDIF()
IF (BUILD_ALBANY_SFAD12)
  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad12"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )
ENDIF()

if (CTEST_DO_SUBMIT)
  ctest_submit (PARTS Build
    RETURN_VALUE  S_HAD_ERROR
    )

  if (S_HAD_ERROR)
    message ("Cannot submit Albany build results!")
  endif ()

endif ()

if (HAD_ERROR)
  message ("Cannot build Albany!")
endif ()

if (BUILD_LIBS_NUM_ERRORS GREATER 0)
  message ("Encountered build errors in Albany build. Exiting!")
endif ()
  
#
# Run Albany tests
#

set (CTEST_TEST_TIMEOUT 1500)
IF (BUILD_ALBANY_SFAD4)
  CTEST_TEST (
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad4"
    RETURN_VALUE HAD_ERROR)
ENDIF()
IF (BUILD_ALBANY_SFAD6)
  CTEST_TEST (
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad6"
    RETURN_VALUE HAD_ERROR)
ENDIF()
IF (BUILD_ALBANY_SFAD8)
  CTEST_TEST (
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad"
    RETURN_VALUE HAD_ERROR)
ENDIF()
IF (BUILD_ALBANY_SFAD12)
  CTEST_TEST (
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad12"
    RETURN_VALUE HAD_ERROR)
ENDIF()

if (CTEST_DO_SUBMIT)
  ctest_submit (PARTS Test RETURN_VALUE S_HAD_ERROR)

  if (S_HAD_ERROR)
    message ("Cannot submit Albany test results!")
  endif ()
endif ()

