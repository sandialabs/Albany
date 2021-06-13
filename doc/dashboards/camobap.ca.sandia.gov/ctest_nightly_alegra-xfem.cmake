
cmake_minimum_required (VERSION 2.8)
set (CTEST_DO_SUBMIT ON)
set (CTEST_TEST_TYPE Nightly)

# What to build and test
set (DOWNLOAD TRUE)
set (BUILD_ALBANY TRUE)
set (BUILD_ALBANY_NOEPETRA FALSE)
set (BUILD_ALBFUNCTOR_OPENMP FALSE)
set (BUILD_CISM_PISCEES_EPETRA FALSE)
set (BUILD_ALBFUNCTOR_OPENMP FALSE)

# Begin User inputs:
set (CTEST_SITE "camobap.ca.sandia.gov" ) # generally the output of hostname
set (CTEST_DASHBOARD_ROOT "$ENV{TEST_DIRECTORY}" ) # writable path
set (CTEST_SCRIPT_DIRECTORY "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set (CTEST_CMAKE_GENERATOR "Unix Makefiles" ) # What is your compilation apps ?
IF (BUILD_ALBANY_FPE) 
set (CTEST_BUILD_CONFIGURATION Debug) # What type of build do you want ?
ELSE()
set (CTEST_BUILD_CONFIGURATION Release) # What type of build do you want ?
ENDIF() 

set (INITIAL_LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})

set (CTEST_PROJECT_NAME "Albany" )
set (CTEST_SOURCE_NAME repos)
set (CTEST_BUILD_NAME "fedora34-gcc11.1.1-${CTEST_BUILD_CONFIGURATION}-alegra-xfem")
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
set (CTEST_BUILD_FLAGS "-j16")

set (CTEST_DROP_METHOD "https")


if (CTEST_DROP_METHOD STREQUAL "https")
  set(CTEST_DROP_METHOD "https")
  set (CTEST_PROJECT_NAME "Albany")
  set(CTEST_DROP_SITE "sems-cdash-son.sandia.gov")
  set(CTEST_DROP_LOCATION "/cdash/submit.php?project=Albany")
  set(CTEST_DROP_SITE_CDASH TRUE)
endif ()

find_program (CTEST_GIT_COMMAND NAMES git)
find_program (CTEST_SVN_COMMAND NAMES svn)

set (alegra-xfem_REPOSITORY_LOCATION git@cee-gitlab.sandia.gov:ikalash/alegra-xfem.git)

if (CLEAN_BUILD)
  # Initial cache info
  set (CACHE_CONTENTS "
  SITE:STRING=${CTEST_SITE}
  CMAKE_BUILD_TYPE:STRING=Release
  CMAKE_GENERATOR:INTERNAL=${CTEST_CMAKE_GENERATOR}
  BUILD_TESTING:BOOL=OFF
  PRODUCT_REPO:STRING=${alegra-xfem_REPOSITORY_LOCATION}
  " )

  ctest_empty_binary_directory( "${CTEST_BINARY_DIRECTORY}" )
  file(WRITE "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt" "${CACHE_CONTENTS}")
endif ()

if (DOWNLOAD)

  set (CTEST_CHECKOUT_COMMAND)
  set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
  #
  # Get alegra-xfem
  #

  if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/alegra-xfem")
    execute_process (COMMAND "${CTEST_GIT_COMMAND}" 
      clone ${alegra-xfem_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/alegra-xfem
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
    
    message(STATUS "out: ${_out}")
    message(STATUS "err: ${_err}")
    message(STATUS "res: ${HAD_ERROR}")
    if (HAD_ERROR)
      message(FATAL_ERROR "Cannot clone alegra-xfem repository!")
    endif ()
  endif ()

  set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")


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
  # Update alegra-xfem
  #

  set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
  CTEST_UPDATE(SOURCE "${CTEST_SOURCE_DIRECTORY}/alegra-xfem" RETURN_VALUE count)
  message("Found ${count} changed files")

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Update
      RETURN_VALUE  HAD_ERROR
      )

    if (HAD_ERROR)
      message(FATAL_ERROR "Cannot update alegra-xfem repository!")
    endif ()
  endif ()

  if (count LESS 0)
    message(FATAL_ERROR "Cannot update alegra-xfem!")
  endif ()

endif ()



if (BUILD_ALBANY)

  # Builds everything!
  #

  set (TRILINSTALLDIR "/nightlyAlbanyTests/Results/Trilinos-extended-sts/build/install")

  set (CONFIGURE_OPTIONS
    "-DTRILINOS_PATH:FILEPATH=${TRILINSTALLDIR}"
    "-DCMAKE_CXX_FLAGS:STRING='-std=gnu++11 -fext-numeric-literals'"
    "-DENABLE_DEBUG_OUTPUT:BOOL=FALSE"
    "-DCMAKE_BUILD_TYPE:STRING=DEBUG"
    "-DBUILD_SHARED_LIBS:BOOL=ON"
    "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF"
    "-DENABLE_ST_LONG_DOUBLE=ON"
    "-DENABLE_ST_FLOAT128=ON") 
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/alegra-xfem-build")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/alegra-xfem-build)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/alegra-xfem-build"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/alegra-xfem/code/TrilinosTSQR"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit alegra-xfem configure results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot configure alegra-xfem build!")
  endif ()

  #
  # Build alegra-xfem
  #

  set (CTEST_BUILD_TARGET all)
  #set (CTEST_BUILD_TARGET install)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/alegra-xfem-build"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Build
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit alegra-xfem build results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot build alegra-xfem!")
  endif ()

  if (BUILD_LIBS_NUM_ERRORS GREATER 0)
    message(FATAL_ERROR "Encountered build errors in alegra-xfem build. Exiting!")
  endif ()

  #
  # Run alegra-xfem tests
  #
  
  set (CTEST_TEST_TIMEOUT 600)

  #  Over-write default limit for output posted to CDash site
  set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE 5000000)
  set(CTEST_CUSTOM_MAXIMUM_FAILED_TEST_OUTPUT_SIZE 5000000)

  CTEST_TEST(
    BUILD "${CTEST_BINARY_DIRECTORY}/alegra-xfem-build"
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
      message(FATAL_ERROR "Cannot submit alegra-xfem test results!")
    endif ()
  endif ()

  #if (HAD_ERROR)
  #	message(FATAL_ERROR "Some alegra-xfem tests failed.")
  #endif ()

endif ()

