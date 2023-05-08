#cmake_minimum_required (VERSION 2.8)
set (CTEST_DO_SUBMIT ON)
set (CTEST_TEST_TYPE Nightly)

# What to build and test
set (CLEAN_BUILD FALSE)
set (DOWNLOAD_COMPASS TRUE)
set (BUILD_COMPASS TRUE)

# Begin User inputs:
set (CTEST_SITE "pm-cpu" ) # generally the output of hostname
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
set (TRILINOS_INSTALL "${CTEST_BINARY_DIRECTORY}/TrilinosSerialInstallGcc")
set (ALBANY_INSTALL "${CTEST_BINARY_DIRECTORY}/AlbanySerialInstallGcc")
execute_process(COMMAND grep "Trilinos_C_COMPILER " ${TRILINOS_INSTALL}/lib/cmake/Trilinos/TrilinosConfig.cmake
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

set (CTEST_BUILD_NAME "compass-${osname}-${osrel}-${COMPILER}-${COMPILER_VERSION}-${CTEST_CONFIGURATION}-serial")

set (CTEST_NIGHTLY_START_TIME "01:00:00 UTC")

set (CTEST_BUILD_COMMAND "bash ${CTEST_DASHBOARD_ROOT}/compass_build.sh")
set (CTEST_TEST_COMMAND "bash ${CTEST_DASHBOARD_ROOT}/compass_test.sh")

set (CTEST_COMMAND "ctest -D ${CTEST_TEST_TYPE}")
set (CTEST_BUILD_FLAGS "-j128")

find_program (CTEST_GIT_COMMAND NAMES git)

set (Albany_REPOSITORY_LOCATION git@github.com:sandialabs/Albany.git)
set (Trilinos_REPOSITORY_LOCATION git@github.com:trilinos/Trilinos.git)
set (Compass_REPOSITORY_LOCATION git@github.com:MPAS-Dev/compass.git)

if (CLEAN_BUILD)
  # Initial cache info
  set (CACHE_CONTENTS "
  SITE:STRING=${CTEST_SITE}
  CMAKE_TYPE:STRING=Release
  CMAKE_GENERATOR:INTERNAL=${CTEST_CMAKE_GENERATOR}
  TESTING:BOOL=OFF
  PRODUCT_REPO:STRING=${Compass_REPOSITORY_LOCATION}
  " )

  ctest_empty_binary_directory( "${CTEST_BINARY_DIRECTORY}" )
  file(WRITE "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt" "${CACHE_CONTENTS}")
endif ()

if (DOWNLOAD_COMPASS)

  set (CTEST_CHECKOUT_COMMAND)
  set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
  
  #
  # Get Compass
  #

  if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/compass")
    execute_process (COMMAND "${CTEST_GIT_COMMAND}" 
      clone ${Compass_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/compass
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
    
    message(STATUS "out: ${_out}")
    message(STATUS "err: ${_err}")
    message(STATUS "res: ${HAD_ERROR}")
    if (HAD_ERROR)
      message(FATAL_ERROR "Cannot clone Compass repository!")
    endif ()
  endif ()

  set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
  
  # Pull the repo
  execute_process (COMMAND "${CTEST_GIT_COMMAND}" pull
      WORKING_DIRECTORY ${CTEST_SOURCE_DIRECTORY}/compass
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
  message(STATUS "Output of Compass pull: ${_out}")
  message(STATUS "Text sent to standard error stream: ${_err}")
  message(STATUS "command result status: ${HAD_ERROR}")
  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot pull Compass!")
  endif ()

endif ()


ctest_start(${CTEST_TEST_TYPE})


if (BUILD_COMPASS)

  #
  # Build compass python environment
  #

  MESSAGE("\nBuilding compass with script '${CTEST_BUILD_COMMAND}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_SOURCE_DIRECTORY}/compass"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Build
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit Compass build results!")
    endif ()

  endif ()

  if (HAD_ERROR)
    message ("Cannot build Compass!")
  endif ()

  if (BUILD_LIBS_NUM_ERRORS GREATER 0)
    message ("Encountered build errors in Compass build. Exiting!")
  endif ()

  #
  # Run compass landice tests
  #
  if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/compass_tests")
    file (MAKE_DIRECTORY "${CTEST_SOURCE_DIRECTORY}/compass_tests")
  endif ()

  #  Over-write default limit for output posted to CDash site
  set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE 5000000)
  set(CTEST_CUSTOM_MAXIMUM_FAILED_TEST_OUTPUT_SIZE 5000000)

  set (CTEST_TEST_TIMEOUT 2400)

  CTEST_TEST(
    BUILD "${CTEST_SOURCE_DIRECTORY}/compass_tests"
    RETURN_VALUE  HAD_ERROR
  )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Test
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit Compass test results!")
    endif ()
  endif ()
endif ()


