cmake_minimum_required (VERSION 2.8)

if (0)
  set (CTEST_DO_SUBMIT ON)
  set (CTEST_TEST_TYPE Nightly)

  # What to build and test
  set (DOWNLOAD TRUE)
  # See if we can get away with this for speed, at least until we get onto a
  # machine that can support a lengthy nightly.
  set (CLEAN_BUILD FALSE)
  set (BUILD_SCOREC TRUE)
  set (BUILD_TRILINOS TRUE)
  set (BUILD_PERIDIGM TRUE)
  set (BUILD_ALB32 TRUE)
  set (BUILD_ALB64 FALSE)
  set (BUILD_TRILINOSCLANG11 TRUE)
  set (BUILD_ALB64CLANG11 TRUE)
  set (BUILD_ALBFUNCTOR TRUE)
  set (BUILD_INTEL_TRILINOS TRUE)
  set (BUILD_INTEL_ALBANY TRUE)
else ()
  # This block is for testing. Set "if (1)" to "if (0)", and then freely mess
  # around with the settings in this block.
  set (CTEST_DO_SUBMIT OFF)
  set (CTEST_TEST_TYPE Experimental)

  # What to build and test
  set (DOWNLOAD FALSE)
  # See if we can get away with this for speed, at least until we get onto a
  # machine that can support a lengthy nightly.
  set (CLEAN_BUILD FALSE)
  set (BUILD_SCOREC TRUE)
  set (BUILD_TRILINOS TRUE)
  set (BUILD_PERIDIGM TRUE)
  set (BUILD_ALB32 TRUE)
  set (BUILD_ALB64 FALSE)
  set (BUILD_TRILINOSCLANG11 TRUE)
  set (BUILD_ALB64CLANG11 TRUE)
  set (BUILD_ALBFUNCTOR TRUE)
  set (BUILD_INTEL_TRILINOS TRUE)
  set (BUILD_INTEL_ALBANY TRUE)
endif ()

set (extra_cxx_flags "")

# Begin User inputs:
set (CTEST_SITE "cee-compute011.sandia.gov" ) # generally the output of hostname
set (CTEST_DASHBOARD_ROOT "$ENV{TEST_DIRECTORY}" ) # writable path
set (CTEST_SCRIPT_DIRECTORY "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
# where the scripts live in the repo
set (CTEST_REPO_SCRIPT_DIRECTORY "${CTEST_SCRIPT_DIRECTORY}/repos/Albany/doc/dashboards/cee-compute011" ) 
set (CTEST_CMAKE_GENERATOR "Unix Makefiles" ) # What is your compilation apps ?
set (CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?

set (INITIAL_LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})
set (PATH $ENV{PATH})

set (CTEST_PROJECT_NAME "Albany" )
set (CTEST_SOURCE_NAME repos)
set (CTEST_BUILD_NAME "linux-gcc-${CTEST_BUILD_CONFIGURATION}")
set (CTEST_BINARY_NAME build)

set (PREFIX_DIR /projects/albany)
set (GCC_MPI_DIR /sierra/sntools/SDK/mpi/openmpi/1.8.8-gcc-5.2.0-RHEL6)
set (INTEL_DIR /sierra/sntools/SDK/compilers/intel/composer_xe_2016.3.210)

#set (BOOST_ROOT /projects/albany/nightly)
set (BOOST_ROOT /projects/albany)

set (INTEL_MPI_DIR /sierra/sntools/SDK/mpi/openmpi/1.8.8-intel-16.0-2016.3.210-RHEL6)
#set (MKL_PATH /sierra/sntools/SDK/compilers/intel)
set (MKL_PATH /sierra/sntools/SDK/compilers/intel/composer_xe_2016.3.210)

set (USE_LAME OFF)
set (LAME_INC_DIR "/projects/sierra/linux_rh6/install/master/lame/include\;/projects/sierra/linux_rh6/install/master/Sierra/sierra_util/include\;/projects/sierra/linux_rh6/install/master/stk/stk_expreval/include\;/projects/sierra/linux_rh6/install/master/utility/include\;/projects/sierra/linux_rh6/install/master/Sierra/include")
set (LAME_LIB_DIR "/projects/sierra/linux_rh6/install/master/lame/lib\;/projects/sierra/linux_rh6/install/master/Sierra/sierra_util/lib\;/projects/sierra/linux_rh6/install/master/stk/stk_expreval/lib\;/projects/sierra/linux_rh6/install/master/utility/lib\;/projects/sierra/linux_rh6/install/master/Sierra/lib")
set (LAME_LIBRARIES "sierra_util_diag\;sierra_util_events\;sierra_util_user_input_function\;sierra_util_domain\;sierra_util_sctl\;stk_expreval\;utility\;sierra\;dataManager\;audit\;sierraparser")
set (MATH_TOOLKIT_INC_DIR
  "/projects/sierra/linux_rh6/install/master/math_toolkit/include")
set (MATH_TOOLKIT_LIB_DIR
  "/projects/sierra/linux_rh6/install/master/math_toolkit/lib")

set (CTEST_SOURCE_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_SOURCE_NAME}")
set (CTEST_BINARY_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_BINARY_NAME}")

IF (CLEAN_BUILD)
  IF(EXISTS "${CTEST_BINARY_DIRECTORY}" )
    FILE(REMOVE_RECURSE "${CTEST_BINARY_DIRECTORY}")
  ENDIF()
ENDIF()

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
set (CTEST_BUILD_FLAGS "-j16")

set (CTEST_DROP_METHOD "http")

if (CTEST_DROP_METHOD STREQUAL "http")
  set (CTEST_DROP_SITE "cdash.sandia.gov")
  set (CTEST_PROJECT_NAME "Albany")
  set (CTEST_DROP_LOCATION "/CDash-2-3-0/submit.php?project=Albany")
  set (CTEST_TRIGGER_SITE "")
  set (CTEST_DROP_SITE_CDASH TRUE)
endif ()

find_program (CTEST_GIT_COMMAND NAMES git)
find_program (CTEST_SVN_COMMAND NAMES svn)

#set (Trilinos_REPOSITORY_LOCATION https://github.com/trilinos/Trilinos.git)
set (Trilinos_REPOSITORY_LOCATION git@github.com:trilinos/Trilinos.git)
set (SCOREC_REPOSITORY_LOCATION git@github.com:SCOREC/core.git)
set (Albany_REPOSITORY_LOCATION git@github.com:gahansen/Albany.git)
#set (Peridigm_REPOSITORY_LOCATION https://github.com/peridigm/peridigm) #ssh://software.sandia.gov/git/peridigm)
set (Peridigm_REPOSITORY_LOCATION git@github.com:peridigm/peridigm) #ssh://software.sandia.gov/git/peridigm)

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
  #
  # Get the internal Trilinos repo
  #

  set (CTEST_CHECKOUT_COMMAND)

  if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/Trilinos")
    execute_process (COMMAND "${CTEST_GIT_COMMAND}" 
      clone ${Trilinos_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/Trilinos
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
    message(STATUS "out: ${_out}")
    message(STATUS "err: ${_err}")
    message(STATUS "res: ${HAD_ERROR}")
    if (HAD_ERROR)
      message(FATAL_ERROR "Cannot clone Trilinos repository!")
    endif ()
  endif ()

  set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")

  #
  # Get the SCOREC repo
  #

  if (BUILD_SCOREC AND (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/Trilinos/SCOREC"))
    #  execute_process (COMMAND "${CTEST_SVN_COMMAND}" 
    #    checkout ${SCOREC_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/Trilinos/SCOREC
    #    OUTPUT_VARIABLE _out
    #    ERROR_VARIABLE _err
    #    RESULT_VARIABLE HAD_ERROR)
    execute_process (COMMAND "${CTEST_GIT_COMMAND}" 
      clone ${SCOREC_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/Trilinos/SCOREC
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
    
    message(STATUS "out: ${_out}")
    message(STATUS "err: ${_err}")
    message(STATUS "res: ${HAD_ERROR}")
    if (HAD_ERROR)
      message ("Cannot checkout SCOREC repository!")
      set (BUILD_SCOREC FALSE)
    endif ()
  endif ()

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

  # Get Peridigm. Nonfatal if error.
  if (BUILD_PERIDIGM AND (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/Peridigm"))
    execute_process (COMMAND ${CTEST_GIT_COMMAND}
      clone ${Peridigm_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/Peridigm
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
    message(STATUS "out: ${_out}")
    message(STATUS "err: ${_err}")
    message(STATUS "res: ${HAD_ERROR}")
    if (HAD_ERROR)
      message (FATAL_ERROR "Cannot clone Peridigm repository.")
      set (BUILD_PERIDIGM FALSE)
    endif ()    
  endif ()
endif ()

ctest_start(${CTEST_TEST_TYPE})

#
# Send the project structure to CDash
#

if (CTEST_DO_SUBMIT)
  ctest_submit (FILES "${CTEST_SCRIPT_DIRECTORY}/Project.xml"
    RETURN_VALUE HAD_ERROR)

  if (HAD_ERROR)
    message ("Cannot submit Albany Project.xml!")
  endif ()
endif ()

if (DOWNLOAD)

  #
  # Update Trilinos
  #

  set_property (GLOBAL PROPERTY SubProject Trilinos)
  set_property (GLOBAL PROPERTY Label Trilinos)

  ctest_update(SOURCE "${CTEST_SOURCE_DIRECTORY}/Trilinos" RETURN_VALUE count)
  message("Found ${count} changed files")

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Update
      RETURN_VALUE  HAD_ERROR
      )

    if (HAD_ERROR)
      message(FATAL_ERROR "Cannot update Trilinos!")
    endif ()
  endif ()

  if (count LESS 0)
    message(FATAL_ERROR "Cannot update Trilinos!")
  endif ()


  #
  # Update the SCOREC repo
  #
  if (BUILD_SCOREC)
    set_property (GLOBAL PROPERTY SubProject SCOREC)
    set_property (GLOBAL PROPERTY Label SCOREC)

    #set (CTEST_UPDATE_COMMAND "${CTEST_SVN_COMMAND}")
    set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
    ctest_update(SOURCE "${CTEST_SOURCE_DIRECTORY}/Trilinos/SCOREC" RETURN_VALUE count)
    message("Found ${count} changed files")

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Update RETURN_VALUE  HAD_ERROR)

      if (HAD_ERROR)
        message ("Cannot update SCOREC!")
        set (BUILD_SCOREC FALSE)
      endif ()
    endif ()

    if (count LESS 0)
      message ("Cannot update SCOREC!")
      set (BUILD_SCOREC FALSE)
    endif ()
  endif ()

  #
  # Update Albany 
  #

  set_property (GLOBAL PROPERTY SubProject Albany32Bit)
  set_property (GLOBAL PROPERTY Label Albany32Bit)

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

  # Peridigm
  if (BUILD_PERIDIGM)
    set_property (GLOBAL PROPERTY SubProject Peridigm)
    set_property (GLOBAL PROPERTY Label Peridigm)

    set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
    ctest_update (SOURCE "${CTEST_SOURCE_DIRECTORY}/Peridigm" RETURN_VALUE count)
    message ("Found ${count} changed files")
    if (count LESS 0)
      set (BUILD_PERIDIGM FALSE)
    endif ()

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Update RETURN_VALUE HAD_ERROR)
    endif ()

    message ("After downloading, BUILD_PERIDIGM = ${BUILD_PERIDIGM}")
  endif ()

endif ()

if (BUILD_TRILINOS)
  INCLUDE(${CTEST_REPO_SCRIPT_DIRECTORY}/trilinos_macro.cmake)
  do_trilinos()
endif (BUILD_TRILINOS)

if (BUILD_PERIDIGM)
  INCLUDE(${CTEST_REPO_SCRIPT_DIRECTORY}/peridigm_macro.cmake)
  do_peridigm()
endif (BUILD_PERIDIGM)

if (BUILD_ALB32)
  INCLUDE(${CTEST_REPO_SCRIPT_DIRECTORY}/alb32_macro.cmake)
  do_alb32()
endif (BUILD_ALB32)

#
# Configure the Albany build using GO = long
#

if (BUILD_ALB64)
  INCLUDE(${CTEST_REPO_SCRIPT_DIRECTORY}/alb64_macro.cmake)
  do_alb64()
endif (BUILD_ALB64)

# Add the path to Clang libraries needed for the Clang configure, build and sest cycle
#
# Need to add the openmpi libraries at the front of LD_LIBRARY_PATH
#

set (ENV{LD_LIBRARY_PATH} 
  ${PREFIX_DIR}/clang/lib:${INITIAL_LD_LIBRARY_PATH}
  )

if (BUILD_TRILINOSCLANG11)
  INCLUDE(${CTEST_REPO_SCRIPT_DIRECTORY}/trilinosclang11_macro.cmake)
  do_trilinosclang11()
endif (BUILD_TRILINOSCLANG11)

#
# Configure the Albany Clang build using GO = long
#

if (BUILD_ALB64CLANG11)
  INCLUDE(${CTEST_REPO_SCRIPT_DIRECTORY}/alb64clang11_macro.cmake)
  do_alb64clang11()
endif (BUILD_ALB64CLANG11)

if (BUILD_ALBFUNCTOR)
  INCLUDE(${CTEST_REPO_SCRIPT_DIRECTORY}/albfunctor_macro.cmake)
  do_albfunctor()
endif (BUILD_ALBFUNCTOR)

if (BUILD_INTEL_TRILINOS)
  INCLUDE(${CTEST_REPO_SCRIPT_DIRECTORY}/intel_macro.cmake)
   do_intel()
endif (BUILD_INTEL_TRILINOS)
