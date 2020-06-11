
cmake_minimum_required (VERSION 2.8)
set (CTEST_DO_SUBMIT ON)
set (CTEST_TEST_TYPE Nightly)

# What to build and test
set (DOWNLOAD FALSE)
set (BUILD_ALBANY FALSE)
set (BUILD_ALBANY_NOEPETRA FALSE)
set (BUILD_ALBFUNCTOR_OPENMP TRUE)

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
set (CTEST_BUILD_NAME "fedora32-gcc10.1.1-${CTEST_BUILD_CONFIGURATION}-Openmp-Albany")
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
  set (CTEST_DROP_SITE "cdash.sandia.gov")
  set (CTEST_PROJECT_NAME "Albany")
  set (CTEST_DROP_LOCATION "/CDash-2-3-0/submit.php?project=Albany")
  set (CTEST_TRIGGER_SITE "")
  set (CTEST_DROP_SITE_CDASH TRUE)
endif ()

find_program (CTEST_GIT_COMMAND NAMES git)
find_program (CTEST_SVN_COMMAND NAMES svn)

set (Albany_REPOSITORY_LOCATION git@github.com:SNLComputation/Albany.git)
set (cism-piscees_REPOSITORY_LOCATION  git@github.com:E3SM-Project/cism-piscees.git)

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

if (DOWNLOAD)

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


if (BUILD_ALBFUNCTOR_OPENMP)
  # ALBANY_KOKKOS_UNDER_DEVELOPMENT build with OpenMP KokkosNode

  set (TRILINSTALLDIR "/nightlyAlbanyTests/Results/Trilinos/build-openmp/install")

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${TRILINSTALLDIR}"
    "-DENABLE_LANDICE:BOOL=ON"
    "-DENABLE_UNIT_TESTS:BOOL=ON"
    "-DENABLE_ALBANY_EPETRA:BOOL=ON"
    "-DSEACAS_EPU=/nightlyAlbanyTests/Results/Trilinos/build/install/bin/epu"
    "-DSEACAS_DECOMP=/nightlyAlbanyTests/Results/Trilinos/build/install/bin/decomp"
    "-DSEACAS_EXODIFF=/nightlyAlbanyTests/Results/Trilinos/build/install/bin/exodiff"
    "-DSEACAS_ALGEBRA=/nightlyAlbanyTests/Results/Trilinos/build/install/bin/algebra"
    "-DENABLE_CHECK_FPE:BOOL=OFF"
    "-DENABLE_MPAS_INTERFACE:BOOL=ON"
    "-DENABLE_CISM_INTERFACE:BOOL=OFF"
    "-DCISM_INCLUDE_DIR:FILEPATH=${CTEST_SOURCE_DIRECTORY}/cism-piscees/libdycore"
    "-DENABLE_KOKKOS_UNDER_DEVELOPMENT:BOOL=ON"
    "-DENABLE_DAKOTA_RESTART_EXAMPLES=OFF"
    "-DENABLE_SLFAD:BOOL=OFF"
    "-DENABLE_64BIT_INT:BOOL=OFF"
    "-DALBANY_MPI_EXEC_TRAILING_OPTIONS='--map-by core'")
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/IKTAlbanyFunctorOpenMP")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/IKTAlbanyFunctorOpenMP)
  endif ()

  CTEST_CONFIGURE (
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbanyFunctorOpenMP"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    APPEND)

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure RETURN_VALUE S_HAD_ERROR)
    
    if (S_HAD_ERROR)
      message ("Cannot submit Albany configure results!")
      set (BUILD_ALBFUNCTOR_OPENMP FALSE)
    endif ()
  endif ()

  if (HAD_ERROR)
    message ("Cannot configure Albany build!")
    set (BUILD_ALBFUNCTOR_OPENMP FALSE)
  endif ()

  if (BUILD_ALBFUNCTOR_OPENMP)
    set (CTEST_BUILD_TARGET all)

    message ("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

    CTEST_BUILD (
      BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbanyFunctorOpenMP"
      RETURN_VALUE HAD_ERROR
      NUMBER_ERRORS BUILD_LIBS_NUM_ERRORS
      APPEND)

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Build
        RETURN_VALUE S_HAD_ERROR)

      if (S_HAD_ERROR)
        message ("Cannot submit Albany build results!")
        set (BUILD_ALBFUNCTOR_OPENMP FALSE)
      endif ()
    endif ()

    if (HAD_ERROR)
      message ("Cannot build Albany!")
      set (BUILD_ALBFUNCTOR_OPENMP FALSE)
    endif ()

    if (BUILD_LIBS_NUM_ERRORS GREATER 0)
      message ("Encountered build errors in Albany build.")
      set (BUILD_ALBFUNCTOR_OPENMP FALSE)
    endif ()
  endif ()

  if (BUILD_ALBFUNCTOR_OPENMP)
    set (CTEST_TEST_TIMEOUT 2400)
    CTEST_TEST (
      BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbanyFunctorOpenMP"
      RETURN_VALUE HAD_ERROR)

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Test RETURN_VALUE S_HAD_ERROR)

      if (S_HAD_ERROR)
        message ("Cannot submit Albany test results!")
      endif ()
    endif ()
  endif ()
endif ()

