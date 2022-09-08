cmake_minimum_required (VERSION 2.8)
set (CTEST_DO_SUBMIT ON)
set (CTEST_TEST_TYPE Nightly)

# What to build and test
set (DOWNLOAD_TRILINOS FALSE)
set (DOWNLOAD_ALBANY TRUE)
set (CLEAN_BUILD FALSE) 
set (BUILD_TRILINOS FALSE)
set (BUILD_ALBANY TRUE)
set (BUILD_CISM_PISCEES FALSE)

# Begin User inputs:
set (CTEST_SITE "cori.nersc.gov" ) # generally the output of hostname
set (CTEST_DASHBOARD_ROOT "$ENV{TEST_DIRECTORY}" ) # writable path
set (CTEST_SCRIPT_DIRECTORY "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set (CTEST_CMAKE_GENERATOR "Unix Makefiles" ) # What is your compilation apps ?
set (CTEST_CONFIGURATION  Release) # What type of build do you want ?

set (INITIAL_LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})

set (CTEST_PROJECT_NAME "Albany" )
set (CTEST_SOURCE_NAME repos)
set (CTEST_BUILD_NAME "cori-Albany")
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
set (CTEST_CMAKE_COMMAND "${PREFIX_DIR}/bin/cmake")
set (CTEST_COMMAND "${PREFIX_DIR}/bin/ctest -D ${CTEST_TEST_TYPE}")
set (CTEST_FLAGS "-j16")
set (CTEST_BUILD_FLAGS "-j16")

set (CTEST_DROP_METHOD "https")

find_program (CTEST_GIT_COMMAND NAMES git)
find_program (CTEST_SVN_COMMAND NAMES svn)

set (Albany_REPOSITORY_LOCATION git@github.com:sandialabs/Albany.git)
set (Trilinos_REPOSITORY_LOCATION git@github.com:trilinos/Trilinos.git)
set (cism-piscees_REPOSITORY_LOCATION  git@github.com:E3SM-Project/cism-piscees.git)


set (BOOST_DIR /project/projectdirs/piscees/tpl/boost_1_55_0) 
#set (NETCDF_DIR /opt/cray/pe/netcdf-hdf5parallel/4.4.0/GNU/5.1) 

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
      clone ${Albany_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/Albany
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

  #
  # Get cism-piscees
  #

  if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/cism-piscees")
    execute_process (COMMAND "${CTEST_GIT_COMMAND}"
      clone ${cism-piscees_REPOSITORY_LOCATION} -b ali_interface ${CTEST_SOURCE_DIRECTORY}/cism-piscees
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
    message(STATUS "out: ${_out}")
    message(STATUS "err: ${_err}")
    message(STATUS "res: ${HAD_ERROR}")
    if (HAD_ERROR)
      message(FATAL_ERROR "Cannot clone cism-piscees repository!")
    endif ()
  endif ()

  set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")


endif ()


ctest_start(${CTEST_TEST_TYPE})

if (BUILD_ALBANY)

  # Configure the Albany build 
  #

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:FILEPATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF"
    "-DENABLE_DEMO_PDES=OFF" 
    "-DENABLE_LANDICE:BOOL=ON"
    "-DENABLE_UNIT_TESTS:BOOL=OFF"
    "-DENABLE_FAD_TYPE:STRING=SFad"
    "-DALBANY_SFAD_SIZE=16"
    "-DENABLE_MPAS_INTERFACE=ON"
    "-DENABLE_CISM_INTERFACE=ON"
    "-DAlbany_BUILD_STATIC_EXE:BOOL=OFF"
    "-DDISABLE_ALBANY_TESTS:BOOL=ON"
    "-DCMAKE_INSTALL_PREFIX:BOOL=${CTEST_BINARY_DIRECTORY}/AlbanyInstall"
    "-DCISM_INCLUDE_DIR:FILEPATH=${CTEST_SOURCE_DIRECTORY}/cism-piscees/libdycore"
    "-DCMAKE_EXE_LINKER_FLAGS:STRING='-Wl,-zmuldefs'"
    "-DCMAKE_C_FLAGS:STRING='-O3'"
    "-DCMAKE_CXX_FLAGS:STRING='-O3 -std=c++1y'"
    "-DCMAKE_VERBOSE_MAKEFILE=ON" 
    "-DCMAKE_BUILD_TYPE:STRING=RELEASE"
    )
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/AlbBuild")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/AlbBuild)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuild"
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

  #set (CTEST_BUILD_TARGET all)
  set (CTEST_BUILD_TARGET install)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuild"
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

  CTEST_TEST(
	  BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuild"
#                  PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
#                  INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
#    NUMBER_FAILED  TEST_NUM_FAILED
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

  if (HAD_ERROR)
    message(FATAL_ERROR "Some Albany tests failed.")
  endif ()

endif()
