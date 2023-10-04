
#cmake_minimum_required (VERSION 2.8)
set (CTEST_DO_SUBMIT ON)
set (CTEST_TEST_TYPE Nightly)
SET(CTEST_BUILD_OPTION "$ENV{BUILD_OPTION}")
set (CTEST_CONFIGURATION  Release) # What type of build do you want ?

# What to build and test
set (DOWNLOAD_ALBANY FALSE) 
set (CTEST_DASHBOARD_ROOT "$ENV{TEST_DIRECTORY}" ) # writable path
set (CTEST_BINARY_NAME build)
set (CTEST_BINARY_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_BINARY_NAME}")

execute_process(COMMAND bash delete_txt_files.sh 
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set (TRILINSTALLDIR "${CTEST_BINARY_DIRECTORY}/TrilinosInstall") 
execute_process(COMMAND grep "Trilinos_C_COMPILER " ${TRILINSTALLDIR}/lib64/cmake/Trilinos/TrilinosConfig.cmake
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		RESULT_VARIABLE MPICC_RESULT
		OUTPUT_FILE "mpicc.txt")
execute_process(COMMAND bash get_mpicc.sh 
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		RESULT_VARIABLE GET_MPICC_RESULT)
execute_process(COMMAND cat mpicc.txt 
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		RESULT_VARIABLE GET_MPICC_RESULT
		OUTPUT_VARIABLE MPICC
		OUTPUT_STRIP_TRAILING_WHITESPACE)
#message("IKT mpicc = " ${MPICC}) 
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
execute_process(COMMAND grep "Trilinos_CXX_COMPILER " ${TRILINSTALLDIR}/lib64/cmake/Trilinos/TrilinosConfig.cmake
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		RESULT_VARIABLE MPICXX_RESULT
		OUTPUT_FILE "mpicxx.txt")
execute_process(COMMAND bash get_mpicxx.sh 
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		RESULT_VARIABLE GET_MPICXX_RESULT)
execute_process(COMMAND cat mpicxx.txt 
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		RESULT_VARIABLE GET_MPICXX_RESULT
		OUTPUT_VARIABLE NVCC
		OUTPUT_STRIP_TRAILING_WHITESPACE)
#message("IKT nvcc = " ${NVCC}) 
execute_process(COMMAND ${NVCC} --version 
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		RESULT_VARIABLE NVCC_RESULT
		OUTPUT_FILE "cuda.txt")
execute_process(COMMAND bash get_cuda_version.sh 
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		RESULT_VARIABLE CUDA_VERSION_RESULT
		OUTPUT_VARIABLE CUDA_VERSION
		OUTPUT_STRIP_TRAILING_WHITESPACE)
#message("IKT cuda version = " ${CUDA_VERSION})

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

set (CTEST_BUILD_NAME "Albany-${osname}-${osrel}-${COMPILER}-${COMPILER_VERSION}-${CTEST_CONFIGURATION}-${CUDA_VERSION}-${CTEST_BUILD_OPTION}")
set (CTEST_NAME "Albany-${osname}-${osrel}-${COMPILER}-${COMPILER_VERSION}-${CTEST_CONFIGURATION}-${CUDA_VERSION}-${CTEST_BUILD_OPTION}")


if (1)
  # What to build and test
  IF(CTEST_BUILD_OPTION MATCHES "sfad6")
    set (BUILD_ALBANY_SFAD6 TRUE)
    set (BUILD_ALBANY_SFAD12 FALSE)
    set (BUILD_ALBANY_SFAD24 FALSE)
    set (SFAD_SIZE 6) 
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "sfad12")
    set (BUILD_ALBANY_SFAD6 FALSE)
    set (BUILD_ALBANY_SFAD12 TRUE)
    set (BUILD_ALBANY_SFAD24 FALSE)
    set (SFAD_SIZE 12) 
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "sfad24")
    set (BUILD_ALBANY_SFAD6 FALSE)
    set (BUILD_ALBANY_SFAD12 FALSE)
    set (BUILD_ALBANY_SFAD24 TRUE)
    set (SFAD_SIZE 24) 
  ENDIF()
ENDIF()


# Begin User inputs:
set (CTEST_SITE "weaver.sandia.gov" ) # generally the output of hostname
set (CTEST_SCRIPT_DIRECTORY "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set (CTEST_CMAKE_GENERATOR "Unix Makefiles" ) # What is your compilation apps ?

set (INITIAL_LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})

set (CTEST_PROJECT_NAME "Albany" )
set (CTEST_SOURCE_NAME repos)

set (CTEST_SOURCE_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_SOURCE_NAME}")

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
set (CTEST_BUILD_FLAGS "-j40")

set (CTEST_DROP_METHOD "https")

if (CTEST_DROP_METHOD STREQUAL "https")
  set (CTEST_DROP_SITE "sems-cdash-son.sandia.gov")
  set (CTEST_PROJECT_NAME "Albany")
  set (CTEST_DROP_LOCATION "/cdash/submit.php?project=Albany")
  set (CTEST_TRIGGER_SITE "")
  set (CTEST_DROP_SITE_CDASH TRUE)
endif ()

find_program (CTEST_GIT_COMMAND NAMES git)

set (Albany_REPOSITORY_LOCATION git@github.com:sandialabs/Albany.git)
set (Trilinos_REPOSITORY_LOCATION git@github.com:trilinos/Trilinos.git)

#set (NVCC_WRAPPER "$ENV{jenkins_trilinos_dir}/packages/kokkos/config/nvcc_wrapper")
set (NVCC_WRAPPER ${CTEST_SCRIPT_DIRECTORY}/nvcc_wrapper_volta)
set (CUDA_MANAGED_FORCE_DEVICE_ALLOC 1)
set( CUDA_LAUNCH_BLOCKING 1)
set( OPENMPI_DIR $ENV{OPENMPI_BIN})

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


# Configure the Albany build 
#

set (CONFIGURE_OPTIONS
  CDASH-ALBANY-FILE.TXT
  )
  
if (BUILD_ALBANY_SFAD6)

  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad6")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/AlbBuildSFad6)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad6"
    SOURCE "/home/projects/albany/nightlyCDashWeaver/repos/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    )
ENDIF()

if (BUILD_ALBANY_SFAD12)
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad12")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/AlbBuildSFad12)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad12"
    SOURCE "/home/projects/albany/nightlyCDashWeaver/repos/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    )
ENDIF()

if (BUILD_ALBANY_SFAD24)
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad24")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/AlbBuildSFad24)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad24"
    SOURCE "/home/projects/albany/nightlyCDashWeaver/repos/Albany"
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


IF (BUILD_ALBANY_SFAD6)
  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad6"
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
IF (BUILD_ALBANY_SFAD24)
  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad24"
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

#  Over-write default limit for output posted to CDash site
set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE 5000000)
set(CTEST_CUSTOM_MAXIMUM_FAILED_TEST_OUTPUT_SIZE 5000000)

set (CTEST_TEST_TIMEOUT 1500)

IF (BUILD_ALBANY_SFAD6)
  CTEST_TEST (
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad6"
    RETURN_VALUE HAD_ERROR)
ENDIF()
IF (BUILD_ALBANY_SFAD12)
  CTEST_TEST (
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad12"
    RETURN_VALUE HAD_ERROR)
ENDIF()
IF (BUILD_ALBANY_SFAD24)
  CTEST_TEST (
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildSFad24"
    RETURN_VALUE HAD_ERROR)
ENDIF()

if (CTEST_DO_SUBMIT)
  ctest_submit (PARTS Test RETURN_VALUE S_HAD_ERROR)

  if (S_HAD_ERROR)
    message ("Cannot submit Albany test results!")
  endif ()
endif ()

