
#cmake_minimum_required (VERSION 2.8)
set (CTEST_DO_SUBMIT ON)
set (CTEST_TEST_TYPE Nightly)

set (DEPLOY_DIR "$ENV{DEPLOY_DIR}")

# What to build and test
set (CLEAN_BUILD FALSE)

set (CTEST_BUILD_OR_TEST "$ENV{BUILD_OR_TEST}")
if(CTEST_BUILD_OR_TEST MATCHES "build")
  set (DOWNLOAD_TRILINOS TRUE)
  set (BUILD_TRILINOS TRUE)
  set (TEST_TRILINOS FALSE)
else()
  set (DOWNLOAD_TRILINOS FALSE)
  set (BUILD_TRILINOS FALSE)
  set (TEST_TRILINOS TRUE)
endif()

# Begin User inputs:
set (CTEST_SITE "frontier" ) # generally the output of hostname
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

execute_process(COMMAND bash delete_txt_files.sh 
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(MPICC $ENV{MPICH_DIR}/bin/mpicc)
#message("IKT MPICC = " ${MPICC}) 
execute_process(COMMAND ${MPICC} -dumpversion 
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		RESULT_VARIABLE COMPILER_VERSION_RESULT
		OUTPUT_VARIABLE COMPILER_VERSION
		OUTPUT_STRIP_TRAILING_WHITESPACE)
#message("IKT compiler version = " ${COMPILER_VERSION})
execute_process(COMMAND ${MPICC} --version 
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		RESULT_VARIABLE COMPILER_RESULT
		OUTPUT_FILE "compiler.txt")
execute_process(COMMAND bash process_compiler.sh 
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		RESULT_VARIABLE CHANGE_COMPILER_RESULT
		OUTPUT_VARIABLE COMPILER
		OUTPUT_STRIP_TRAILING_WHITESPACE)
#message("IKT compiler = " ${COMPILER})


find_program(UNAME NAMES uname)
macro(getuname name flag)
  exec_program("${UNAME}" ARGS "${flag}" OUTPUT_VARIABLE "${name}")
endmacro(getuname)

getuname(osname -s)
getuname(osrel  -r)
getuname(cpu    -m)

#message("IKT osname = " ${osname}) 
#message("IKT osrel = " ${osrel}) 
#message("IKT cpu = " ${cpu}) 

set (CTEST_BUILD_NAME "Trilinos-${osname}-${osrel}-${COMPILER}-${COMPILER_VERSION}-${CTEST_CONFIGURATION}-rocm")

set (CTEST_NIGHTLY_START_TIME "01:00:00 UTC")
set (CTEST_CMAKE_COMMAND "cmake")
set (CTEST_COMMAND "ctest -D ${CTEST_TEST_TYPE}")
set (CTEST_BUILD_FLAGS "-j128")

find_program (CTEST_GIT_COMMAND NAMES git)

set (Albany_REPOSITORY_LOCATION https://github.com/sandialabs/Albany.git)
set (Trilinos_REPOSITORY_LOCATION https://github.com/trilinos/Trilinos.git)

set (CRAYPE_LINK_TYPE dynamic)

set (CRAY_XPMEM_POST_LINK_OPTS $ENV{CRAY_XPMEM_POST_LINK_OPTS})
set (PE_MPICH_GTL_DIR_amd_gfx90a $ENV{PE_MPICH_GTL_DIR_amd_gfx90a})
set (PE_MPICH_GTL_LIBS_amd_gfx90a $ENV{PE_MPICH_GTL_LIBS_amd_gfx90a})

set (MPICH_DIR $ENV{MPICH_DIR})
set (CRAY_LIBSCI_PREFIX_DIR $ENV{CRAY_LIBSCI_PREFIX_DIR})
set (BOOST_ROOT $ENV{BOOST_ROOT})
set (NETCDF_DIR $ENV{NETCDF_DIR})
set (PNETCDF_DIR $ENV{PNETCDF_DIR})
set (HDF5_DIR $ENV{HDF5_DIR})
 
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

set (CTEST_CHECKOUT_COMMAND)

#
# Get Trilinos
#
  
if (DOWNLOAD_TRILINOS)
  execute_process (COMMAND "${CTEST_GIT_COMMAND}" checkout develop
    WORKING_DIRECTORY ${CTEST_SOURCE_DIRECTORY}/Trilinos
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
    RESULT_VARIABLE HAD_ERROR)
  message(STATUS "out: ${_out}")
  message(STATUS "err: ${_err}")
  message(STATUS "res: ${HAD_ERROR}")
  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot checkout develop!")
  endif ()

  # Pull the repo
  execute_process (COMMAND "${CTEST_GIT_COMMAND}" pull
      WORKING_DIRECTORY ${CTEST_SOURCE_DIRECTORY}/Trilinos
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
  message(STATUS "Output of Trilinos pull: ${_out}")
  message(STATUS "Text sent to standard error stream: ${_err}")
  message(STATUS "command result status: ${HAD_ERROR}")
  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot pull Trilinos!")
  endif ()
endif()

ctest_start(${CTEST_TEST_TYPE})

# Set the common Trilinos config options & build Trilinos
# 
if (BUILD_TRILINOS) 
  message ("ctest state: BUILD_TRILINOS")
  #
  # Configure the Trilinos build
  #
  set(INSTALL_DIR /lustre/orion/cli193/proj-shared/automated_testing/rocm/builds/TrilinosInstall)

  set (CONFIGURE_OPTIONS
    CDASH-TRILINOS-FILE.TXT
  )

  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/TriBuildRocmGcc")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/TriBuildRocmGcc)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuildRocmGcc"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Trilinos"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    )

  message ("ctest state: Configure trilinos complete")

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit Trilinos configure results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message ("Cannot configure Trilinos build!")
  endif ()

  #
  # Build the rest of Trilinos and install everything
  #

  #set (CTEST_BUILD_TARGET all)
  set (CTEST_BUILD_TARGET install)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuildRocmGcc"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Build
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit Trilinos build results!")
    endif ()

  endif ()

  if (HAD_ERROR)
    message ("Cannot build Trilinos!")
  endif ()

  if (BUILD_LIBS_NUM_ERRORS GREATER 0)
    message ("Encountered build errors in Trilinos build. Exiting!")
  endif ()
endif()

if (TEST_TRILINOS)
  #
  # Run Trilinos tests
  #
  set (CTEST_TEST_TIMEOUT 500)

  CTEST_TEST(
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuildRocmGcc"
    RETURN_VALUE  HAD_ERROR
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Test
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit Trilinos test results!")
    endif ()
  endif ()
endif()
