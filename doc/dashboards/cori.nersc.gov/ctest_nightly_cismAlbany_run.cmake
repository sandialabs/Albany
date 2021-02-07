cmake_minimum_required (VERSION 2.8)
set (CTEST_DO_SUBMIT ON)
set (CTEST_TEST_TYPE Nightly)

# What to build and test
set (DOWNLOAD_TRILINOS FALSE)
set (DOWNLOAD_ALBANY FALSE)
set (CLEAN_BUILD FALSE) 
set (BUILD_TRILINOS FALSE)
set (BUILD_ALBANY FALSE)
set (BUILD_CISM_PISCEES FALSE)
set (RUN_CISM_PISCEES TRUE)

# Begin User inputs:
set (CTEST_SITE "cori.nersc.gov" ) # generally the output of hostname
set (CTEST_DASHBOARD_ROOT "$ENV{TEST_DIRECTORY}" ) # writable path
set (CTEST_SCRIPT_DIRECTORY "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set (CTEST_CMAKE_GENERATOR "Unix Makefiles" ) # What is your compilation apps ?
set (CTEST_CONFIGURATION  Release) # What type of build do you want ?

set (INITIAL_LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})

set (CTEST_PROJECT_NAME "Albany" )
set (CTEST_SOURCE_NAME repos)
set (CTEST_BUILD_NAME "cori-CISM-Albany")
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

set (Albany_REPOSITORY_LOCATION git@github.com:SNLComputation/Albany.git)
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



ctest_start(${CTEST_TEST_TYPE})

if (BUILD_CISM_PISCEES)

  # Configure the CISM-Albany build 
  #

  set (CONFIGURE_OPTIONS
    "-Wno-dev"
    "-DCISM_MPI_MODE:BOOL=ON"
    "-DCISM_SERIAL_MODE:BOOL=OFF"
    "-DCISM_BUILD_CISM_DRIVER:BOOL=ON"
    "-DCISM_USE_GPTL_INSTRUMENTATION:BOOL=OFF"
    "-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON"
    #
    "-DCISM_USE_TRILINOS:BOOL=ON"
    "-DCISM_TRILINOS_DIR=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "-DALBANY_LANDICE_DYCORE:BOOL=ON"
    "-DALBANY_LANDICE_CTEST:BOOL=ON"
    "-DCISM_ALBANY_DIR=${CTEST_BINARY_DIRECTORY}/AlbanyFELIXInstall"
    "-DCISM_NETCDF_DIR=$ENV{NETCDF_DIR}"
    "-DPYTHON_EXE=/opt/python/2.7.15.6/bin/python" 
    #
    "-DCMAKE_CXX_COMPILER=CC"
    "-DCMAKE_C_COMPILER=cc"
    "-DCMAKE_Fortran_COMPILER=ftn"
    #
    "-DCMAKE_EXE_LINKER_FLAGS:STRING='-Wl,-zmuldefs'"
    "-DBUILD_SHARED_LIBS:BOOL=ON"
    "-DCISM_STATIC_LINKING:BOOL=OFF"
    "-DCISM_Fortran_FLAGS='-ffree-line-length-none'" 
    "-DCISM_GNU:BOOL=ON"
    "-DCMAKE_C_FLAGS:STRING='-O3 -std=c++1y'"
    "-DCMAKE_CXX_FLAGS:STRING='-O3 -std=c++1y'"
  )
 
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/CoriCismAlbany")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/CoriCismAlbany)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/CoriCismAlbany"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/cism-piscees"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit CISM-Albany configure results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot configure CISM-Albany build!")
  endif ()

  #
  # Build CISM-Albany
  #

  set (CTEST_TARGET all)

  MESSAGE("\nBuilding target: '${CTEST_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/CoriCismAlbany"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  LIBS_NUM_ERRORS
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Build
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit CISM-Albany build results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot build CISM-Albany!")
  endif ()

  if (LIBS_NUM_ERRORS GREATER 0)
    message(FATAL_ERROR "Encountered build errors in CISM-Albany build. Exiting!")
  endif ()
endif ()

IF(RUN_CISM_PISCEES) 
  #
  # Run CISM-Albany tests
  #

  CTEST_TEST(
    BUILD "${CTEST_BINARY_DIRECTORY}/CoriCismAlbany"
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
      message(FATAL_ERROR "Cannot submit CISM-Albany test results!")
    endif ()
  endif ()

  if (HAD_ERROR)
  	message(FATAL_ERROR "Some CISM-Albany tests failed.")
  endif ()

endif ()
