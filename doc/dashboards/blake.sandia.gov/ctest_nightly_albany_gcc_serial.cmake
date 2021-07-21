#cmake_minimum_required (VERSION 2.8)
set (CTEST_DO_SUBMIT ON)
set (CTEST_TEST_TYPE Nightly)

# What to build and test
set (CLEAN_BUILD TRUE)
set (DOWNLOAD_TRILINOS FALSE)
set (BUILD_TRILINOS_SERIAL FALSE)
set (BUILD_TRILINOS_OPENMP FALSE)
set (DOWNLOAD_ALBANY TRUE) 
set (BUILD_ALBANY_SERIAL TRUE) 
set (BUILD_ALBANY_SERIAL_SFAD FALSE) 
set (BUILD_ALBANY_OPENMP FALSE) 


# Begin User inputs:
set (CTEST_SITE "blake.sandia.gov" ) # generally the output of hostname
set (CTEST_DASHBOARD_ROOT "$ENV{TEST_DIRECTORY}" ) # writable path
set (CTEST_SCRIPT_DIRECTORY "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set (CTEST_CMAKE_GENERATOR "Unix Makefiles" ) # What is your compilation apps ?
set (CTEST_CONFIGURATION  Release) # What type of build do you want ?

set (INITIAL_LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})

set (CTEST_PROJECT_NAME "Albany" )
set (CTEST_SOURCE_NAME repos)
set (CTEST_BUILD_NAME "blake-serial-Albany-gcc-no-warn")
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
set (CTEST_BUILD_FLAGS "-j48")

find_program (CTEST_GIT_COMMAND NAMES git)

set (Albany_REPOSITORY_LOCATION git@github.com:SNLComputation/Albany.git)
set (Trilinos_REPOSITORY_LOCATION git@github.com:trilinos/Trilinos.git)
set (MPI_PATH $ENV{MPI_ROOT})  
set (MKL_PATH $ENV{MKL_ROOT})  
set (BOOST_PATH $ENV{BOOST_ROOT}) 
set (NETCDF_PATH $ENV{NETCDF_ROOT}) 
set (HDF5_PATH $ENV{HDF5_ROOT})
set (ZLIB_PATH $ENV{ZLIB_ROOT})  

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
  
  # Pull the repo
  execute_process (COMMAND "${CTEST_GIT_COMMAND}" pull
      WORKING_DIRECTORY ${CTEST_SOURCE_DIRECTORY}/Albany
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
  message(STATUS "Output of Albany pull: ${_out}")
  message(STATUS "Text sent to standard error stream: ${_err}")
  message(STATUS "command result status: ${HAD_ERROR}")
  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot pull Albany!")
  endif ()

endif ()


ctest_start(${CTEST_TEST_TYPE})


if (BUILD_ALBANY_SERIAL)

  # Configure the Albany build 
  #

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:FILEPATH=/home/projects/albany/nightlyCDashTrilinosBlake/build/TrilinosSerialInstallGccNoWarn"
    "-DENABLE_LANDICE:BOOL=ON"
    "-DENABLE_DEMO_PDES:BOOL=ON"
    "-DENABLE_KOKKOS_UNDER_DEVELOPMENT:BOOL=ON"
    "-DALBANY_CTEST_TIMEOUT=1800"
    "-DENABLE_CHECK_FPE:BOOL=OFF"
    "-DALBANY_MPI_EXEC_TRAILING_OPTIONS='--map-by ppr:1:core:pe=4'"
    )
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/AlbBuildSerialGccNoWarn")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/AlbBuildSerialGccNoWarn)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildSerialGccNoWarn"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit Albany configure results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message ("Cannot configure Albany build!")
  endif ()

  #
  # Build the rest of Albany and install everything
  #

  set (CTEST_BUILD_TARGET all)
  #set (CTEST_BUILD_TARGET install)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildSerialGccNoWarn"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

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
  #  Over-write default limit for output posted to CDash site
  set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE 5000000)
  set(CTEST_CUSTOM_MAXIMUM_FAILED_TEST_OUTPUT_SIZE 5000000)

  set (CTEST_TEST_TIMEOUT 1800)

  CTEST_TEST(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildSerialGccNoWarn"
    RETURN_VALUE  HAD_ERROR
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Test
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit Albany test results!")
    endif ()
  endif ()
endif ()
