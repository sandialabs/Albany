
# Begin User inputs:
set (CTEST_SITE "algol.ca.sandia.gov" ) # generally the output of hostname
set (CTEST_DASHBOARD_ROOT "$ENV{TEST_DIRECTORY}" ) # writable path
set (CTEST_SCRIPT_DIRECTORY "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set (CTEST_CMAKE_GENERATOR "Unix Makefiles" ) # What is your compilation apps ?
set (CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?

set (INITIAL_LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})

set (CTEST_PROJECT_NAME "Albany" )
set (CTEST_SOURCE_NAME repos)
set (CTEST_BUILD_NAME "linux-gcc-${CTEST_BUILD_CONFIGURATION}")
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

set (CTEST_NIGHTLY_START_TIME "00:00:00 UTC")
set (CTEST_CMAKE_COMMAND "${PREFIX_DIR}/bin/cmake")
set (CTEST_COMMAND "${PREFIX_DIR}/bin/ctest -D ${CTEST_TEST_TYPE}")
set (CTEST_BUILD_FLAGS "-j72")

set (CTEST_DROP_METHOD "http")

if (CTEST_DROP_METHOD STREQUAL "http")
  set (CTEST_DROP_SITE "cdash.sandia.gov")
  set (CTEST_PROJECT_NAME "Albany")
  set (CTEST_DROP_LOCATION "/CDash-2-3-0/submit.php?project=Albany")
  set (CTEST_TRIGGER_SITE "")
  set (CTEST_DROP_SITE_CDASH TRUE)
endif ()

find_program (CTEST_GIT_COMMAND NAMES git)

set (Albany_REPOSITORY_LOCATION git@github.com:gahansen/Albany.git)

if (CLEAN_BUILD)
  # Initial cache info
  set (CACHE_CONTENTS "
  SITE:STRING=${CTEST_SITE}
  CMAKE_BUILD_TYPE:STRING=Release
  CMAKE_GENERATOR:INTERNAL=${CTEST_CMAKE_GENERATOR}
  BUILD_TESTING:BOOL=OFF
  PRODUCT_REPO:STRING=${Albany_REPOSITORY_LOCATION}
  " )

  ctest_empty_binary_directory( "${CTEST_BINARY_DIRECTORY}" )
  file(WRITE "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt" "${CACHE_CONTENTS}")
endif ()

ctest_start(${CTEST_TEST_TYPE})

#
# Send the project structure to CDash
#

if (CTEST_DO_SUBMIT)
  ctest_submit (FILES "${CTEST_SCRIPT_DIRECTORY}/Project.xml"
    RETURN_VALUE  HAD_ERROR
    )

  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot submit Albany Project.xml!")
  endif ()
endif ()

if (DOWNLOAD)

  #
  # Update Albany 
  #

  set_property (GLOBAL PROPERTY SubProject LCMAlbanyReleaseAlgol)
  set_property (GLOBAL PROPERTY Label LCMAlbanyReleaseAlgol)

  set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
  CTEST_UPDATE(SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany" RETURN_VALUE count)
  message("Found ${count} changed files")

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Update
      RETURN_VALUE  HAD_ERROR
      )

    if (HAD_ERROR)
      message(FATAL_ERROR "Cannot update Albany repository!")
    endif ()
  endif ()

  if (count LESS 0)
    message(FATAL_ERROR "Cannot update Albany!")
  endif ()

endif ()

if (BUILD_ALB_LCM_RELEASE)

  # Configure Albany 
  #

  set_property (GLOBAL PROPERTY SubProject LCMAlbanyReleaseAlgol)
  set_property (GLOBAL PROPERTY Label LCMAlbanyReleaseAlgol)

  set (TRILINSTALLDIR "/home/lcm/LCM/trilinos-install-serial-gcc-release")

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:FILEPATH=${TRILINSTALLDIR}"
    "-DCMAKE_CXX_FLAGS:STRING='-msse3'"
    "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF"
    "-DENABLE_LCM:BOOL=ON"
    "-DENABLE_ATO:BOOL=OFF"
    "-DENABLE_QCAD:BOOL=OFF"
    "-DENABLE_MOR:BOOL=OFF"
    "-DENABLE_SG:BOOL=OFF"
    "-DENABLE_ENSEMBLE:BOOL=OFF"
    "-DENABLE_FELIX:BOOL=OFF"
    "-DENABLE_LAME:BOOL=OFF"
    "-DENABLE_LAMENT:BOOL=OFF"
    "-DENABLE_CHECK_FPE:BOOL=OFF"
    "-DENABLE_FLUSH_DENORMALS:BOOL=ON"
    "-DENABLE_KOKKOS_UNDER_DEVELOPMENT:BOOL=OFF"
    "-DALBANY_ENABLE_FORTRAN:BOOL=OFF"
    "-DENABLE_SLFAD:BOOL=OFF"
  ) 
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/LCMAlbanyReleaseAlgol")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/LCMAlbanyReleaseAlgol)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/LCMAlbanyReleaseAlgol"
    SOURCE "/home/lcm/LCM/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit Albany configure results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot configure Albany build!")
  endif ()

  #
  # Build Albany
  #

  set (CTEST_BUILD_TARGET all)
  #set (CTEST_BUILD_TARGET install)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/LCMAlbanyReleaseAlgol"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Build
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit Albany build results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot build Albany!")
  endif ()

  if (BUILD_LIBS_NUM_ERRORS GREATER 0)
    message(FATAL_ERROR "Encountered build errors in Albany build. Exiting!")
  endif ()

  #
  # Run Albany tests
  #

  CTEST_TEST(
    BUILD "${CTEST_BINARY_DIRECTORY}/LCMAlbanyReleaseAlgol"
    #              PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
    #              INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
    #NUMBER_FAILED  TEST_NUM_FAILED
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

  #if (HAD_ERROR)
  #	message(FATAL_ERROR "Some Albany tests failed.")
  #endif ()

endif ()
