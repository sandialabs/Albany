cmake_minimum_required (VERSION 2.8)

SET(CTEST_DO_SUBMIT "$ENV{DO_SUBMIT}")
SET(CTEST_TEST_TYPE "$ENV{TEST_TYPE}")
SET(CTEST_BUILD_OPTION "$ENV{BUILD_OPTION}")

execute_process(COMMAND bash $ENV{SCRIPT_DIRECTORY}/delete_txt_files.sh
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
message("IKT mpicc = " $ENV{OPENMPI_ROOT}/bin/mpicc) 
execute_process(COMMAND $ENV{OPENMPI_ROOT}/bin/mpicc -dumpversion
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE COMPILER_VERSION_RESULT
                OUTPUT_VARIABLE COMPILER_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)
#message("IKT compiler version = " ${COMPILER_VERSION})
execute_process(COMMAND $ENV{OPENMPI_ROOT}/bin/mpicc --version
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE COMPILER_RESULT
                OUTPUT_FILE "compiler.txt")
execute_process(COMMAND bash $ENV{SCRIPT_DIRECTORY}/process_compiler.sh
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE CHANGE_COMPILER_RESULT
                OUTPUT_VARIABLE COMPILER
                OUTPUT_STRIP_TRAILING_WHITESPACE)
message("IKT compiler = " ${COMPILER})
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

if (1)
  # What to build and test
  IF(CTEST_BUILD_OPTION MATCHES "base-trilinos") 
    set (DOWNLOAD_TRILINOS FALSE)
    set (DOWNLOAD_ALBANY FALSE)
    set (BUILD_TRILINOS TRUE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_STRING "Trilinos")  
  ENDIF() 
  IF(CTEST_BUILD_OPTION MATCHES "base-albany") 
    set (DOWNLOAD_TRILINOS FALSE)
    set (DOWNLOAD_ALBANY FALSE)
    set (BUILD_TRILINOS FALSE)
    set (BUILD_ALB64 TRUE) 
    set (BUILD_STRING "Albany")  
  ENDIF() 
  set (CLEAN_BUILD TRUE)
  set (CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?
  IF(CTEST_BUILD_OPTION MATCHES "debug-trilinos")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_TRILINOSDBG TRUE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_ALB64DBG FALSE)
    set (BUILD_TRILINOSCLANG FALSE)
    set (BUILD_ALB64CLANG FALSE)
    set (BUILD_TRILINOSCLANGDBG FALSE)
    set (BUILD_ALB64CLANGDBG FALSE)
    set (BUILD_INTEL_TRILINOS FALSE)
    set (BUILD_INTEL_ALBANY FALSE)
    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
    set (BUILD_STRING "Trilinos")  
  ELSE()
    set (BUILD_TRILINOSDBG FALSE)
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "debug-albany")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_TRILINOSDBG FALSE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_ALB64DBG TRUE)
    set (BUILD_TRILINOSCLANG FALSE)
    set (BUILD_ALB64CLANG FALSE)
    set (BUILD_TRILINOSCLANGDBG FALSE)
    set (BUILD_ALB64CLANGDBG FALSE)
    set (BUILD_INTEL_TRILINOS FALSE)
    set (BUILD_INTEL_ALBANY FALSE)
    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
    set (BUILD_STRING "Albany")  
  ELSE()
    set (BUILD_ALB64DBG FALSE)
  ENDIF() 
  IF(CTEST_BUILD_OPTION MATCHES "clang-trilinos")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_TRILINOSCLANG TRUE)
    set (BUILD_ALB64CLANG FALSE)
    set (BUILD_TRILINOSCLANGDBG FALSE)
    set (BUILD_ALB64CLANGDBG FALSE)
    set (BUILD_INTEL_TRILINOS FALSE)
    set (BUILD_INTEL_ALBANY FALSE)
    set (BUILD_STRING "Trilinos")  
    set (CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?
#    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
  ELSE()
    set (BUILD_TRILINOSCLANG FALSE)
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "clang-albany")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_TRILINOSCLANG FALSE)
    set (BUILD_ALB64CLANG TRUE)
    set (BUILD_TRILINOSCLANGDBG FALSE)
    set (BUILD_ALB64CLANGDBG FALSE)
    set (BUILD_INTEL_TRILINOS FALSE)
    set (BUILD_INTEL_ALBANY FALSE)
    set (BUILD_STRING "Albany")  
    set (CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?
#    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
  ELSE()
    set (BUILD_ALB64CLANG FALSE)
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "clangdbg-trilinos")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_TRILINOSCLANG FALSE)
    set (BUILD_ALB64CLANG FALSE)
    set (BUILD_TRILINOSCLANGDBG TRUE)
    set (BUILD_ALB64CLANGDBG FALSE)
    set (BUILD_INTEL_TRILINOS FALSE)
    set (BUILD_INTEL_ALBANY FALSE)
    set (BUILD_STRING "Trilinos")  
    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
#    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
  ELSE()
    set (BUILD_TRILINOSCLANGDBG FALSE)
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "clangdbg-albany")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_TRILINOSCLANG FALSE)
    set (BUILD_ALB64CLANG FALSE)
    set (BUILD_TRILINOSCLANGDBG FALSE)
    set (BUILD_ALB64CLANGDBG TRUE)
    set (BUILD_INTEL_TRILINOS FALSE)
    set (BUILD_INTEL_ALBANY FALSE)
    set (BUILD_STRING "Albany")  
    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
#    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
  ELSE()
    set (BUILD_ALB64CLANGDBG FALSE)
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "intel-trilinos")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_TRILINOSCLANG FALSE)
    set (BUILD_ALB64CLANG FALSE)
    set (BUILD_TRILINOSCLANGDBG FALSE)
    set (BUILD_ALB64CLANGDBG FALSE)
    set (BUILD_INTEL_TRILINOS TRUE)
    set (BUILD_INTEL_ALBANY FALSE)
    set (BUILD_STRING "Trilinos")  
    set (CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?
#    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
  ELSE()
    set (BUILD_INTEL_TRILINOS FALSE)
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "intel-albany")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_TRILINOSCLANG FALSE)
    set (BUILD_ALB64CLANG FALSE)
    set (BUILD_TRILINOSCLANGDBG FALSE)
    set (BUILD_ALB64CLANGDBG FALSE)
    set (BUILD_INTEL_TRILINOS FALSE)
    set (BUILD_INTEL_ALBANY TRUE)
    set (BUILD_STRING "Albany")  
    set (CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?
#    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
  ELSE()
    set (BUILD_INTEL_ALBANY FALSE)
  ENDIF()
else ()
  # This block is for testing. Set "if (1)" to "if (0)", and then freely mess
  # around with the settings in this block.

  # What to build and test
  set (DOWNLOAD_TRILINOS FALSE)
  set (DOWNLOAD_ALBANY FALSE)
  # See if we can get away with this for speed, at least until we get onto a
  # machine that can support a lengthy nightly.
  set (CLEAN_BUILD FALSE)
  set (BUILD_TRILINOS FALSE)
  set (BUILD_ALB64 FALSE)
  set (BUILD_TRILINOSCLANG FALSE)
  set (BUILD_ALB64CLANG FALSE)
  set (BUILD_TRILINOSCLANGDBG FALSE)
  set (BUILD_ALB64CLANGDBG FALSE)
  set (BUILD_INTEL_TRILINOS FALSE)
  set (BUILD_INTEL_ALBANY FALSE)
  set (CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?
#  set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
endif ()

set (extra_cxx_flags "")

# Begin User inputs:
#set (CTEST_SITE "cee-compute011.sandia.gov" ) # generally the output of hostname
SITE_NAME(CTEST_SITE) # directly set CTEST_SITE to the output of `hostname`
set (CTEST_DASHBOARD_ROOT "$ENV{INSTALL_DIRECTORY}" ) # writable path
set (CTEST_SCRATCH_ROOT "$ENV{SCRATCH_DIRECTORY}" ) # writable path
set (CTEST_SCRIPT_ROOT "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
#IKT, 9/26/2022: Use Ninja only for Trilinos builds,  not Albany builds
#IF((CTEST_BUILD_OPTION MATCHES "base-albany") OR (CTEST_BUILD_OPTION MATCHES "debug-albany") OR (CTEST_BUILD_OPTION MATCHES "clang-albany") OR (CTEST_BUILD_OPTION MATCHES "clangdbg-albany") OR (CTEST_BUILD_OPTION MATCHES "intel-albany"))
#  set (CTEST_CMAKE_GENERATOR "Unix Makefiles" ) # What is your compilation apps ?
#ELSE ()
  set (CTEST_CMAKE_GENERATOR "Ninja")
#ENDIF()

set (CTEST_PROJECT_NAME "Albany" )
set (CTEST_SOURCE_NAME repos)
#set (CTEST_BUILD_NAME "linux-gcc-${CTEST_BUILD_CONFIGURATION}")
#set (CTEST_BUILD_NAME "${osname}-${osrel}-${CTEST_BUILD_OPTION}-${CTEST_BUILD_CONFIGURATION}")
#set (CTEST_BUILD_NAME "${osname}-${osrel}-${CTEST_BUILD_OPTION}")
set (CTEST_BUILD_NAME "${BUILD_STRING}-${osname}-${osrel}-${COMPILER}-${COMPILER_VERSION}-${CTEST_BUILD_CONFIGURATION}-Serial")
set (CTEST_BINARY_NAME build)
set (CTEST_INSTALL_NAME test)

#  Over-write default limit for output posted to CDash site
set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE 5000000)
set(CTEST_CUSTOM_MAXIMUM_FAILED_TEST_OUTPUT_SIZE 5000000)

if ((CTEST_BUILD_CONFIGURATION MATCHES "Debug") OR (CTEST_BUILD_OPTION MATCHES "clang-albany"))
# Runs tests longer if in debug mode
   set (CTEST_TEST_TIMEOUT 4200)
else () 
   set (CTEST_TEST_TIMEOUT 900)
endif ()

set (PREFIX_DIR /projects/albany)

set (CTEST_SOURCE_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_SOURCE_NAME}")
# Build all results in a scratch space
set (CTEST_BINARY_DIRECTORY "${CTEST_SCRATCH_ROOT}/${CTEST_BINARY_NAME}")
# Trilinos, etc installed here
set (CTEST_INSTALL_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_INSTALL_NAME}")

if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}")
  file (MAKE_DIRECTORY "${CTEST_SOURCE_DIRECTORY}")
endif ()
if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}")
  file (MAKE_DIRECTORY "${CTEST_BINARY_DIRECTORY}")
endif ()
if (NOT EXISTS "${CTEST_INSTALL_DIRECTORY}")
  file (MAKE_DIRECTORY "${CTEST_INSTALL_DIRECTORY}")
endif ()

# Clean up storage area for nightly testing results
IF (CLEAN_BUILD)
  IF(EXISTS "${CTEST_BINARY_DIRECTORY}/Testing" )
    FILE(REMOVE_RECURSE "${CTEST_BINARY_DIRECTORY}/Testing")
  ENDIF()
ENDIF()

configure_file (${CTEST_SCRIPT_DIRECTORY}/CTestConfig.cmake
  ${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake COPYONLY)

set (CTEST_NIGHTLY_START_TIME "01:00:00 UTC")
set (CTEST_CMAKE_COMMAND "${PREFIX_DIR}/bin/cmake")
set (CTEST_COMMAND "${PREFIX_DIR}/bin/ctest -D ${CTEST_TEST_TYPE}")
#IKT, 9/26/2022: use Ninja only for Trilinos builds, not Albany
#IF((CTEST_BUILD_OPTION MATCHES "base-albany") OR (CTEST_BUILD_OPTION MATCHES "debug-albany") OR (CTEST_BUILD_OPTION MATCHES "clang-albany") OR (CTEST_BUILD_OPTION MATCHES "clangdbg-albany") OR (CTEST_BUILD_OPTION MATCHES "intel-albany"))
#  set (CTEST_BUILD_FLAGS "-j16")
#ELSE()
  #IKT, 4/10/2022: the following is for Ninja build
  set (CTEST_BUILD_FLAGS "${CTEST_BUILD_FLAGS}-k 999999")
#ENDIF()

set (CTEST_DROP_METHOD "https")

if (CTEST_DROP_METHOD STREQUAL "https")
  set (CTEST_DROP_SITE "sems-cdash-son.sandia.gov")
  set (CTEST_PROJECT_NAME "Albany")
  set (CTEST_DROP_LOCATION "/cdash/submit.php?project=Albany")
  set (CTEST_TRIGGER_SITE "")
  set (CTEST_DROP_SITE_CDASH TRUE)
endif ()

find_program (CTEST_GIT_COMMAND NAMES git)
find_program (CTEST_SVN_COMMAND NAMES svn)

set (Trilinos_REPOSITORY_LOCATION git@github.com:trilinos/Trilinos.git)
set (Albany_REPOSITORY_LOCATION git@github.com:sandialabs/Albany.git)

if (DOWNLOAD_TRILINOS)
  #
  # Get the internal Trilinos repo
  #

  set (CTEST_CHECKOUT_COMMAND)

  if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/Trilinos")
    execute_process (COMMAND "${CTEST_GIT_COMMAND}" 
      clone --branch develop ${Trilinos_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/Trilinos
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

  #set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")

ENDIF()

IF (DOWNLOAD_ALBANY) 
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

ENDIF ()

ctest_start(${CTEST_TEST_TYPE})

#
# Send the project structure to CDash
#

if (CTEST_DO_SUBMIT)
  ctest_submit (FILES "${CTEST_SCRIPT_DIRECTORY}/Project.xml"
    RETURN_VALUE HAD_ERROR)

  if (HAD_ERROR)
    message ("Cannot submit Albany Project.xml!")
  endif (HAD_ERROR)
endif (CTEST_DO_SUBMIT)

if (DOWNLOAD_TRILINOS)

  #
  # Update Trilinos
  #

  CTEST_UPDATE(SOURCE "${CTEST_SOURCE_DIRECTORY}/Trilinos" RETURN_VALUE count)
  # assumes that we are already on the desired tracking branch, i.e.,
  # git checkout -b branch --track origin/branch
  message("Found ${count} changed files")

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Update
      RETURN_VALUE  HAD_ERROR
      )

    if (HAD_ERROR)
      message("Cannot update Trilinos!")
    endif ()
  endif ()

  if (count LESS 0)
    message(FATAL_ERROR "Cannot update Trilinos!")
  endif ()

ENDIF()

IF(DOWNLOAD_ALBANY) 
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
      message("Cannot update Albany repository!")
    endif ()
  endif ()

  if (count LESS 0)
    message(FATAL_ERROR "Cannot update Albany!")
  endif ()

endif ()

INCLUDE(${CTEST_SCRIPT_DIRECTORY}/trilinos_macro.cmake)

if (BUILD_INTEL_TRILINOS)
  set(BTYPE "RELEASE") 
  set(INSTALL_LOCATION "${CTEST_INSTALL_DIRECTORY}/TrilinosIntelInstall")
  set (CONF_OPTS
    CDASH-TRILINOS-INTEL-FILE.TXT
  )
  # First argument is the string of the configure options, second is the dashboard target (a name in a string)
  do_trilinos("${CONF_OPTS}" "TrilinosIntel" "${INSTALL_LOCATION}")
  
endif (BUILD_INTEL_TRILINOS)

if (BUILD_TRILINOS OR BUILD_TRILINOSDBG)

  if (BUILD_TRILINOS) 
    set(INSTALL_LOCATION "${CTEST_INSTALL_DIRECTORY}/TrilinosInstall")
    set(BTYPE "RELEASE") 
    set(CCFLAGS "-O3 -march=native -DNDEBUG -Wno-inconsistent-missing-override")
    set(CFLAGS "-O3 -march=native -DNDEBUG")
    set(FFLAGS "-O3 -march=native -DNDEBUG -Wa,-q")
  endif(BUILD_TRILINOS)
  if (BUILD_TRILINOSDBG) 
    set(INSTALL_LOCATION "${CTEST_INSTALL_DIRECTORY}/TrilinosDbg")
    set(BTYPE "DEBUG") 
    set(CCFLAGS "-g -O0 -Wno-inconsistent-missing-override") 
    set(CFLAGS "-g -O0")
    set(FFLAGS "-g -O0 -Wa,-q") 
  endif(BUILD_TRILINOSDBG)
  set (CONF_OPTS
    "-Wno-dev"
    CDASH-TRILINOS-GCC-FILE.TXT
  )

  # First argument is the string of the configure options, second is the dashboard target (a name in a string)
  do_trilinos("${CONF_OPTS}" "Trilinos" "${INSTALL_LOCATION}")

endif (BUILD_TRILINOS OR BUILD_TRILINOSDBG)


if (BUILD_TRILINOSCLANG OR BUILD_TRILINOSCLANGDBG)

  if (BUILD_TRILINOSCLANG) 
    set(INSTALL_LOCATION "${CTEST_INSTALL_DIRECTORY}/TrilinosInstallC11")
    set(BTYPE "RELEASE") 
    set(CCFLAGS "-O3 -march=native -DNDEBUG=1")
    set(CFLAGS "-O3 -march=native -DNDEBUG=1")
    set(FFLAGS "-O3 -march=native -DNDEBUG=1 -Wa,-q")
  endif (BUILD_TRILINOSCLANG) 
  if (BUILD_TRILINOSCLANGDBG) 
    set(INSTALL_LOCATION "${CTEST_INSTALL_DIRECTORY}/TrilinosInstallC11Dbg")
    set(BTYPE "DEBUG")
    set(CCFLAGS "-g -O0")
    set(CFLAGS "-g -O0")
    set(FFLAGS "-g -O0 -Wa,-q")
  endif (BUILD_TRILINOSCLANGDBG) 
  
  set (CONF_OPTS
    "-Wno-dev"
    CDASH-TRILINOS-CLANG-FILE.TXT
  )

  # First argument is the string of the configure options, second is the dashboard target (a name in a string)
  do_trilinos("${CONF_OPTS}" "Trilinos" "${INSTALL_LOCATION}")

endif (BUILD_TRILINOSCLANG OR BUILD_TRILINOSCLANGDBG)

INCLUDE(${CTEST_SCRIPT_DIRECTORY}/albany_macro.cmake)

#
# Configure the Albany build using GO = long
#

if (BUILD_ALB64 OR BUILD_ALB64DBG OR BUILD_ALB64CLANG OR BUILD_ALB64CLANGDBG OR BUILD_INTEL_ALBANY)

if (BUILD_ALB64) 
  set(TRILINSTALLDIR ${CTEST_INSTALL_DIRECTORY}/TrilinosInstall) 
  set(BUILDTYPE "RELEASE")
  set(FPE_CHECK "OFF")
  set(MESH_DEP_ON_SOLN "ON")
  set(MESH_DEP_ON_PARAMS "OFF")
endif(BUILD_ALB64) 
if (BUILD_INTEL_ALBANY)
  set(TRILINSTALLDIR ${CTEST_INSTALL_DIRECTORY}/TrilinosIntelInstall)
  set(BUILDTYPE "RELEASE")
  set(FPE_CHECK "OFF")
  set(MESH_DEP_ON_SOLN "OFF")
  set(MESH_DEP_ON_PARAMS "ON")
endif (BUILD_INTEL_ALBANY)
if (BUILD_ALB64CLANG)
  set(TRILINSTALLDIR ${CTEST_INSTALL_DIRECTORY}/TrilinosInstallC11)
  set(BUILDTYPE "RELEASE")
  set(FPE_CHECK "OFF")
  set(MESH_DEP_ON_SOLN "OFF")
  set(MESH_DEP_ON_PARAMS "OFF")
endif (BUILD_ALB64CLANG)
if (BUILD_ALB64CLANGDBG)
  set(TRILINSTALLDIR ${CTEST_INSTALL_DIRECTORY}/TrilinosInstallC11Dbg) 
  set(BUILDTYPE "DEBUG")
  set(FPE_CHECK "ON")
  set(MESH_DEP_ON_SOLN "OFF")
  set(MESH_DEP_ON_PARAMS "OFF")
endif (BUILD_ALB64CLANGDBG)
if (BUILD_ALB64DBG)
  set(TRILINSTALLDIR ${CTEST_INSTALL_DIRECTORY}/TrilinosDbg)
  set(BUILDTYPE "DEBUG")
  set(FPE_CHECK "ON")
  set(MESH_DEP_ON_SOLN "OFF")
  set(MESH_DEP_ON_PARAMS "OFF")
endif (BUILD_ALB64DBG)

  set (CONF_OPTIONS
    CDASH-ALBANY-FILE.TXT
    )

  # First argument is the string of the configure options, second is the dashboard target (a name in a string)
if (BUILD_ALB64) 
  do_albany("${CONF_OPTIONS}" "Albany64Bit")
endif(BUILD_ALB64) 
if (BUILD_INTEL_ALBANY)
  do_albany("${CONF_OPTIONS}" "AlbanyIntel")
endif (BUILD_INTEL_ALBANY)
if (BUILD_ALB64CLANG)
  do_albany("${CONF_OPTIONS}" "Albany64BitClang")
endif (BUILD_ALB64CLANG)
if (BUILD_ALB64CLANGDBG)
  do_albany("${CONF_OPTIONS}" "Albany64BitClangDbg")
endif (BUILD_ALB64CLANGDBG)
if (BUILD_ALB64DBG)
  do_albany("${CONF_OPTIONS}" "Albany64BitDbg")
endif (BUILD_ALB64DBG)

endif (BUILD_ALB64 OR BUILD_ALB64DBG OR BUILD_ALB64CLANG OR BUILD_ALB64CLANGDBG OR BUILD_INTEL_ALBANY)


