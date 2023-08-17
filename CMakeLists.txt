# When debugging the CMake logic, it can be helpful to
# print all vars with a certain regex, to see what's going on.
# This function does precisely that. If REGEX is not provided,
# it prints ALL cmake vars in scope
function(dump_cmake_variables)
  set (opts)
  set (args1v REGEX)
  set (argsMv)
  cmake_parse_arguments (dcv "${opts}" "${args1v}" "${argsMv}" ${ARGN})

  get_cmake_property(dcv_var_names VARIABLES)

  foreach (var_name IN LISTS dcv_var_names)
    if (dcv_REGEX)
      string(REGEX MATCH ${dcv_REGEX} MATCHES ${var_name})
      if (MATCHES)
        message(STATUS "${var_name}=${${var_name}}")
      endif()
    else ()
      message(STATUS "${var_name}=${${var_name}}")
    endif ()
  endforeach()
endfunction()

##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//


# CMAKE File for Albany building against an installed Trilinos

cmake_minimum_required(VERSION 3.17.0)
include(CMakeDependentOption)

if (NOT DEFINED SET_COMPILERS_AUTOMATICALLY)
  set(SET_COMPILERS_AUTOMATICALLY TRUE)
endif()

# Kokkos recommends to set CMAKE_CXX_EXTENSIONS to OFF
if (NOT DEFINED CMAKE_CXX_EXTENSIONS)
  set(CMAKE_CXX_EXTENSIONS OFF)
ENDIF ()

# Advertise that we support shared libs
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

MESSAGE("ALBANY_TRILINOS_DIR = " ${ALBANY_TRILINOS_DIR}) 

# Get Trilinos as one entity
IF (DEFINED ALBANY_TRILINOS_DIR)
  SET(CMAKE_PREFIX_PATH ${ALBANY_TRILINOS_DIR} ${CMAKE_PREFIX_PATH})
ENDIF()
FIND_PACKAGE(Trilinos 14.1 REQUIRED)

OPTION (ALBANY_SUPPRESS_TRILINOS_WARNINGS "Whether or not Trilinos headers should be treated as 'system' headers (hence, without issuing warnings)" ON)

IF (DEFINED CMAKE_BUILD_TYPE)
   IF ("${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
      MESSAGE("-- CMAKE_BUILD_TYPE=DEBUG...  Debug build is Enabled")
      SET(ALBANY_DEBUG_BUILD ON)
   ELSE()
      MESSAGE("-- CMAKE_BUILD_TYPE=RELEASE...  Debug build is NOT Enabled")
      SET(ALBANY_DEBUG_BUILD OFF)
   ENDIF()
ENDIF()
# Trilinos_BIN_DIRS probably should be defined in the Trilinos config. Until it is, set it here.
# This is needed to find SEACAS tools used during testing (epu, etc).
SET(Trilinos_BIN_DIRS "${Trilinos_DIR}/../../../bin")
SET(Trilinos_LIB_DIRS "${Trilinos_DIR}/../../../lib")

# Set Trilinos commit id
IF(EXISTS "${Trilinos_DIR}/../../../TrilinosRepoVersion.txt")
  FILE(STRINGS "${Trilinos_DIR}/../../../TrilinosRepoVersion.txt" TRILINOS_VERSION_FILE)
  SET(TRILINOS_GIT_COMMIT_ID_LINE_LOC 1)
  LIST(GET TRILINOS_VERSION_FILE ${TRILINOS_GIT_COMMIT_ID_LINE_LOC} TRILINOS_GIT_COMMIT_ID_LINE)
  STRING(REGEX MATCH "^[^ ]+" Trilinos_GIT_COMMIT_ID ${TRILINOS_GIT_COMMIT_ID_LINE})
  SET(ALBANY_TRILINOS_GIT_COMMIT_ID ${Trilinos_GIT_COMMIT_ID})
ENDIF()

MESSAGE("\nFound Trilinos!  Here are the details: ")
MESSAGE("-- Trilinos_DIR = ${Trilinos_DIR}")
MESSAGE("-- Trilinos_CMAKE_BUILD_TYPE = ${Trilinos_CMAKE_BUILD_TYPE}")
MESSAGE("-- Trilinos_VERSION = ${Trilinos_VERSION}")
MESSAGE("-- Trilinos_GIT_COMMIT_ID = ${Trilinos_GIT_COMMIT_ID}")
MESSAGE("-- Trilinos_BIN_DIRS = ${Trilinos_BIN_DIRS}")
MESSAGE("-- Trilinos_LIBRARY_DIRS = ${Trilinos_LIBRARY_DIRS}")
MESSAGE("-- Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
MESSAGE("-- Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
MESSAGE("-- Trilinos_TPL_LIBRARY_DIRS = ${Trilinos_TPL_LIBRARY_DIRS}")
MESSAGE("-- Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
MESSAGE("-- Trilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
MESSAGE("End of Trilinos details\n")

# Compress Trilinos lists that are used
LIST(REMOVE_DUPLICATES Trilinos_INCLUDE_DIRS)
LIST(REMOVE_DUPLICATES Trilinos_TPL_INCLUDE_DIRS)
LIST(REMOVE_DUPLICATES Trilinos_LIBRARIES)
LIST(REMOVE_DUPLICATES Trilinos_TPL_LIBRARIES)

if (SET_COMPILERS_AUTOMATICALLY)
  MESSAGE("Setting and checking of compilers:")
  if (Albany_CROSS_COMPILE)
    # Force the compilers to be the same as Trilinos (GAH experimental)
    # This syntax is needed when cross compiling or the compilers get checked
    # again by the Albany configure, and will probably fail, because the options
    # won't be right and the exes won't run on the host.
    INCLUDE (CMakeForceCompiler)
    SET(CMAKE_SYSTEM_NAME Generic)
    CMAKE_FORCE_CXX_COMPILER(${Trilinos_CXX_COMPILER} Generic)
    CMAKE_FORCE_C_COMPILER(${Trilinos_C_COMPILER} Generic)
    CMAKE_FORCE_Fortran_COMPILER(${Trilinos_Fortran_COMPILER} Generic)
#    SET(CMAKE_SYSTEM_NAME ${Trilinos_SYSTEM_NAME})
#    CMAKE_FORCE_CXX_COMPILER(${Trilinos_CXX_COMPILER} ${Trilinos_CXX_COMPILER_ID})
#    CMAKE_FORCE_C_COMPILER(${Trilinos_C_COMPILER} ${Trilinos_C_COMPILER_ID})
#    CMAKE_FORCE_Fortran_COMPILER(${Trilinos_Fortran_COMPILER} ${Trilinos_Fortran_COMPILER_ID})
#    SET(CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES ${Trilinos_Fortran_IMPLICIT_LINK_LIBRARIES})
  else ()
    set (CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
    set (CMAKE_C_COMPILER ${Trilinos_C_COMPILER})
    set (CMAKE_Fortran_COMPILER ${Trilinos_Fortran_COMPILER})
  endif ()
else()
  # Make sure the compilers match.
  MESSAGE("Checking compilers:")
  IF(NOT ${Trilinos_CXX_COMPILER} STREQUAL ${CMAKE_CXX_COMPILER})
    MESSAGE(FATAL_ERROR "C++ compilers don't match (Trilinos: ${Trilinos_CXX_COMPILER}, ${PROJECT_NAME}: ${CMAKE_CXX_COMPILER}).")
  ENDIF()
  IF(NOT ${Trilinos_C_COMPILER} STREQUAL ${CMAKE_C_COMPILER})
    MESSAGE(FATAL_ERROR "C compilers don't match (Trilinos: ${Trilinos_C_COMPILER}, ${PROJECT_NAME}: ${CMAKE_C_COMPILER}).")
  ENDIF()
  IF(NOT ${Trilinos_Fortran_COMPILER} STREQUAL ${CMAKE_Fortran_COMPILER})
    MESSAGE(FATAL_ERROR "Fortran compilers don't match (Trilinos: ${Trilinos_Fortran_COMPILER}, ${PROJECT_NAME}: ${CMAKE_Fortran_COMPILER}).")
  ENDIF()
  SET(Trilinos_CXX_COMPILER_ID ${CMAKE_CXX_COMPILER_ID})
  SET(Trilinos_C_COMPILER_ID ${CMAKE_C_COMPILER_ID})
  SET(Trilinos_Fortran_COMPILER_ID ${CMAKE_Fortran_COMPILER_ID})
endif()

set (CMAKE_BUILD_TYPE ${Trilinos_CMAKE_BUILD_TYPE} CACHE STRING "Build type for Albany (default to Trilinos)")

project(Albany CXX C)

# Set Albany branch and commit id
EXECUTE_PROCESS(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE ALBANY_GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
EXECUTE_PROCESS(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE ALBANY_GIT_COMMIT_ID
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
MESSAGE("Albany version details:")
MESSAGE("-- ALBANY_GIT_BRANCH = ${ALBANY_GIT_BRANCH}")
MESSAGE("-- ALBANY_GIT_COMMIT_ID = ${ALBANY_GIT_COMMIT_ID}")
MESSAGE("End of Albany version details\n")

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
MESSAGE(" +----------------------------------+")
MESSAGE(" |       Installation details       |")
MESSAGE(" +----------------------------------+")
MESSAGE(" | - Prefix: ${CMAKE_INSTALL_PREFIX}")
MESSAGE(" | - Binaries dir: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}")
MESSAGE(" | - Headers dir: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")
MESSAGE(" | - Libraries dir: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
MESSAGE(" +----------------------------------+")

option (ENABLE_CROSS_COMPILE "Flag to turn on cross-compiling tools" OFF)
if (ENABLE_CROSS_COMPILE)
  set (Albany_CROSS_COMPILE ON)
  message ("Building with cross-compiling tools.")
else ()
  set (Albany_CROSS_COMPILE OFF)
endif ()

# Build Albany as shared libraries if Trilinos was compiled that way
IF(Trilinos_BUILD_SHARED_LIBS)
  SET(Albany_LIBRARY_TYPE SHARED)
  SET(BUILD_SHARED_LIBS ON)
  MESSAGE("-- Building Albany with shared libraries")
ELSE(Trilinos_BUILD_SHARED_LIBS)
  SET(Albany_LIBRARY_TYPE STATIC)
ENDIF(Trilinos_BUILD_SHARED_LIBS)

OPTION(Albany_BUILD_STATIC_EXE "Flag to turn on building of static executables in Albany" OFF)
IF((NOT BUILD_SHARED_LIBS) AND (Albany_BUILD_STATIC_EXE))
  MESSAGE("-- Building Albany with static executables")
ENDIF((NOT BUILD_SHARED_LIBS) AND (Albany_BUILD_STATIC_EXE))

# Option to turn on and off deprecation warnings
OPTION(Albany_DEPRECATED_DECLARATIONS "Flag to turn on warnings for deprecated code" ON)
IF (Albany_DEPRECATED_DECLARATIONS)
  SET(ALBANY_ENABLE_DEPRECATED ON)
  MESSAGE("-- Deprecated code warnings Enabled")
ELSE()
  MESSAGE("-- Deprecated code warnings are NOT Enabled.")
ENDIF()

# AGS: Adding these lines so Ctest can be run to submit to cdash dashboard
#   Uses CTestConfig.cmake file for Cdash info.
ENABLE_TESTING()
option(BUILD_TESTING "" OFF) # Avoid creating folder "Testing" at config time
INCLUDE(CTest)

set(ALBANY_ENABLE_FORTRAN ON CACHE BOOL "enable fortran" )

IF (CMAKE_Fortran_COMPILER AND ALBANY_ENABLE_FORTRAN) # Enable Fortran if it is enabled in Trilinos.
  ENABLE_LANGUAGE(Fortran)
ENDIF()

# Make Albany inherit Trilinos compiler flags. The most important are the
# presence or absence of -NDEBUG (which can affect linking for complicated
# reasons) and -std=c++11.
# GAH - temporarily set Albany flags for clang debugging
#SET(CMAKE_CXX_FLAGS "-ggdb -std=c++11 -O2 -march=native -DNDEBUG")

if (SET_COMPILERS_AUTOMATICALLY)
  SET(CMAKE_AR ${Trilinos_AR})
  SET(CMAKE_LINKER ${Trilinos_LINKER})
endif()

# No longer needed but keep for now as this shows how to silence specific warnings on Intel compiler
#IF(${Trilinos_CXX_COMPILER_ID} MATCHES "Intel")
#  SET(USING_INTEL TRUE)
# On Intel, turn of interprocedural optimization at linking to keep linker from hanging
# This could possibly be removed with further study (GAH)
#  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-ip")
#   SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd2536,2196,858,488,161,47")
#ENDIF()

# Compiler details
SET(ALBANY_CXX_COMPILER_ID ${CMAKE_CXX_COMPILER_ID})
MESSAGE("-- ALBANY_CXX_COMPILER_ID = ${ALBANY_CXX_COMPILER_ID}")
SET(ALBANY_CXX_COMPILER_VERSION ${CMAKE_CXX_COMPILER_VERSION})
MESSAGE("-- ALBANY_CXX_COMPILER_VERSION = ${ALBANY_CXX_COMPILER_VERSION}")
IF(EXISTS "${Trilinos_INCLUDE_DIRS}/KokkosCore_config.h")
  FILE(READ "${Trilinos_INCLUDE_DIRS}/KokkosCore_config.h" KOKKOS_CORE_CONFIG_FILE)
  STRING(REGEX MATCH "#define KOKKOS_ENABLE_OPENMP" ALBANY_ENABLE_OPENMP ${KOKKOS_CORE_CONFIG_FILE})
  IF(ALBANY_ENABLE_OPENMP)
    MESSAGE("-- Kokkos is configured to use OpenMP, Albany will also.")
  ENDIF()
  STRING(REGEX MATCH "#define KOKKOS_ENABLE_CUDA" ALBANY_ENABLE_CUDA ${KOKKOS_CORE_CONFIG_FILE})
  IF(ALBANY_ENABLE_CUDA)
    MESSAGE("-- Kokkos is configured to use CUDA, Albany will also.")
    STRING(REGEX MATCH "#define KOKKOS_COMPILER_CUDA_VERSION ([0-9]*)" _ ${KOKKOS_CORE_CONFIG_FILE})
    SET(ALBANY_CUDA_COMPILER_VERSION "${CMAKE_MATCH_1}")
    MESSAGE("-- ALBANY_CUDA_COMPILER_VERSION = ${ALBANY_CUDA_COMPILER_VERSION}")
  ENDIF()
ELSE()
  MESSAGE(FATAL_ERROR "\nError: ${Trilinos_INCLUDE_DIRS}/KokkosCore_config.h not found!")
ENDIF()
MESSAGE("End of compiler details\n")

## Check for some required Trilinos package dependencies (Piro, Phalanx) ##
MESSAGE("Checking Trilinos build for required and optional packages")

OPTION(ENABLE_ALBANY_EPETRA "Flag to turn on building of code depending on Epetra" ON)

OPTION(ENABLE_ALBANY_PYTHON "Flag to turn on building of code depending on Python" OFF)

###############################
### Check REQUIRED packages ###
###############################

MESSAGE ("\n  **********************************")
MESSAGE ("  *** Required Trilinos Packages ***")
MESSAGE ("  **********************************\n")
# Piro
LIST(FIND Trilinos_PACKAGE_LIST Piro Piro_List_ID)
IF (Piro_List_ID GREATER -1)
  MESSAGE("-- Looking for Piro ...                          Found.")
ELSE()
  MESSAGE("-- Looking for Piro ...                      NOT Found.")
  MESSAGE(FATAL_ERROR "Albany REQUIRES Piro. Please, configure Trilinos with guidance from trilinos-cmake script in doc directory.")
ENDIF()

# Panzer (new blocked data structures use Panzer DOF manager
LIST(FIND Trilinos_PACKAGE_LIST PanzerDofMgr Panzer_List_ID)
IF (Panzer_List_ID GREATER -1)
  MESSAGE("-- Looking for PanzerDofMgr ...                  Found.")
ELSE()
  MESSAGE("-- Looking for PanzerDofMgr ...              NOT Found.")
  MESSAGE(FATAL_ERROR "Albany REQUIRES PanzerDofMgr. Please, configure Trilinos with guidance from trilinos-cmake script in doc directory.")
ENDIF()

# Phalanx
LIST(FIND Trilinos_PACKAGE_LIST Phalanx Phalanx_List_ID)
IF (Phalanx_List_ID GREATER -1)
  MESSAGE("-- Looking for Phalanx ...                       Found.")
ELSE()
  MESSAGE("-- Looking for Phalanx ...                   NOT Found.")
  MESSAGE(FATAL_ERROR "Albany REQUIRES Phalanx. Please, configure Trilinos with guidance from trilinos-cmake script in doc directory.")
ENDIF()

# NOX
LIST(FIND Trilinos_PACKAGE_LIST NOX NOX_List_ID)
IF (NOX_List_ID GREATER -1)
  MESSAGE("-- Looking for NOX ...                           Found.")
ELSE()
  MESSAGE("-- Looking for NOX ...                       NOT Found.")
  MESSAGE(FATAL_ERROR "Albany REQUIRES NOX. Please, configure Trilinos with guidance from trilinos-cmake script in doc directory.")
ENDIF()

# ThyraTpetraAdapters
LIST(FIND Trilinos_PACKAGE_LIST ThyraTpetraAdapters ThyraTpetraAdapters_List_ID)
IF (ThyraTpetraAdapters_List_ID GREATER -1)
  MESSAGE("-- Looking for ThyraTpetraAdapters ...           Found.")
ELSE()
  MESSAGE("-- Looking for ThyraTpetraAdapters ...       NOT Found.")
  MESSAGE(FATAL_ERROR "Albany REQUIRES ThyraTpetraAdapters. Please, configure Trilinos with guidance from trilinos-cmake script in doc directory.")
ENDIF()

# Intrepid2
LIST(FIND Trilinos_PACKAGE_LIST Intrepid2 Intrepid2_List_ID)
IF (Intrepid2_List_ID GREATER -1)
  MESSAGE("-- Looking for Intrepid2 ...                     Found.")
ELSE()
  MESSAGE("-- Looking for Intrepid2 ...                 NOT Found.")
  MESSAGE(FATAL_ERROR "Albany REQUIRES Intrepid2. Please, configure Trilinos with guidance from trilinos-cmake script in doc directory.")
ENDIF()

# Tempus
LIST(FIND Trilinos_PACKAGE_LIST Tempus Tempus_List_ID)
IF (Tempus_List_ID GREATER -1)
  MESSAGE("-- Looking for Tempus ...                        Found.")
  SET(ALBANY_TEMPUS TRUE)
ELSE()
  MESSAGE("-- Looking for Tempus ...                    NOT Found.")
  MESSAGE(FATAL_ERROR "Albany REQUIRES Tempus. Please, configure Trilinos with guidance from trilinos-cmake script in doc directory.")
ENDIF()

# stk
LIST(FIND Trilinos_PACKAGE_LIST STK have_stk_List_ID)
IF (have_stk_List_ID GREATER -1)
  MESSAGE("-- Looking for stk ...                           Found.")

ELSE()
  MESSAGE("-- Looking for stk ...                       NOT Found.")
  MESSAGE(FATAL_ERROR "Albany REQUIRES stk. Please, configure Trilinos with guidance from trilinos-cmake script in doc directory.")
ENDIF()

###############################
### Check OPTIONAL packages ###
###############################

MESSAGE ("\n  **********************************")
MESSAGE ("  *** Optional Trilinos Packages ***")
MESSAGE ("  **********************************\n")
### Check for some OPTIONAL trilinos packages, which set DEFINES in Alany build ###

# set optional dependency on Panzer's ExprEval subpackage
LIST (FIND Trilinos_PACKAGE_LIST PanzerExprEval PanzerExprEval_List_ID)
IF (PanzerExprEval_List_ID GREATER -1)
  MESSAGE("-- Looking for PanzerExprEval ...                Found.")
  SET(ALBANY_PANZER_EXPR_EVAL TRUE)
ELSE ()
  MESSAGE("-- Looking for PanzerExprEval ...            NOT Found.")
  SET(ALBANY_PANZER_EXPR_EVAL FALSE)
ENDIF()

#set optional dependency on STKExprEval
LIST(FIND Trilinos_PACKAGE_LIST STKExprEval STKExprEval_List_ID)
IF (STKExprEval_List_ID GREATER -1)
  MESSAGE("-- Looking for STKExprEval ...                   Found.")
  SET(ALBANY_STK_EXPR_EVAL TRUE)
  add_definitions(-DALBANY_STK_EXPR_EVAL) 
ELSE()
  MESSAGE("-- Looking for STKExprEval ...               NOT Found.")
  SET(ALBANY_STK_EXPR_EVAL FALSE)
ENDIF()

# set optional dependency on Zoltan, for partitioning
# MESSAGE("-- Looking for Zoltan:")
LIST(FIND Trilinos_PACKAGE_LIST Zoltan Zoltan_List_ID)
IF (Zoltan_List_ID GREATER -1)
  SET(ALBANY_ZOLTAN TRUE)
  SET(ALBANY_STK_REBALANCE TRUE)
  MESSAGE("-- Looking for Zoltan ...                        Found.")
ELSE()
  MESSAGE("-- Looking for Zoltan ...                    NOT Found.")
  SET(ALBANY_ZOLTAN FALSE)
  SET(ALBANY_STK_REBALANCE FALSE)
ENDIF()

# set optional dependency on Stokhos, for KL expansions
# MESSAGE("-- Looking for Stokhos:")
LIST(FIND Trilinos_PACKAGE_LIST Stokhos Stokhos_List_ID)
IF (Stokhos_List_ID GREATER -1)
  SET(ALBANY_STOKHOS TRUE)
  MESSAGE("-- Looking for Stokhos ...                       Found.")
ELSE()
  MESSAGE("-- Looking for Stokhos ...                   NOT Found.")
  SET(ALBANY_STOKHOS FALSE)
ENDIF()

# set optional dependency on Belos's Epetra subpackage
IF (EXISTS "${Trilinos_INCLUDE_DIRS}/BelosEpetraAdapter.hpp")
  MESSAGE("-- Looking for BelosEpetra ...                   Found.")
  SET(ALBANY_BELOS_EPETRA TRUE)
ELSE ()
  MESSAGE("-- Looking for BelosEpetra ...               NOT Found.")
  SET(ALBANY_BELOS_EPETRA FALSE)
ENDIF()

# set optional dependency on Amesos2, for solving and preconditioning
LIST(FIND Trilinos_PACKAGE_LIST Amesos2 Amesos2_List_ID)
IF (Amesos2_List_ID GREATER -1)
  MESSAGE("-- Looking for Amesos2 ...                       Found.")
  SET(ALBANY_AMESOS2 TRUE)
  IF (Amesos2_ENABLE_SuperLUDist)
    MESSAGE("-- Looking for Amesos2 SuperLUDist...            Found.")
    SET(ALBANY_AMESOS2_SUPERLUDIST TRUE)
  ELSE()
    MESSAGE("-- Looking for Amesos2 SuperLUDist...        Not Found.")
    SET(ALBANY_AMESOS2_SUPERLUDIST FALSE)
  ENDIF()

  IF (Amesos2_ENABLE_SuperLU)
    MESSAGE("-- Looking for Amesos2 SuperLU...                Found.")
    SET(ALBANY_AMESOS2_SUPERLU TRUE)
  ELSE()
    MESSAGE("-- Looking for Amesos2 SuperLU...            Not Found.")
    SET(ALBANY_AMESOS2_SUPERLU FALSE)
  ENDIF()
  # We can't rely on Amesos2_TPL_LIST any longer; KLU2 and Basker are not
  # TPLs. These will soon be enabled by default in Amesos2. So we'll just
  # assume we have a coarse direct solver.
  SET(ALBANY_AMESOS2_OPERATIONAL_SOLVER TRUE)
ELSE()
  MESSAGE("-- Looking for Amesos2 ...                   NOT Found.")
  SET(ALBANY_AMESOS2 FALSE)
  SET(ALBANY_AMESOS2_OPERATIONAL_SOLVER FALSE)
ENDIF()

# set optional dependency on Ifpack2, for preconditioning
LIST(FIND Trilinos_PACKAGE_LIST Ifpack2 Ifpack2_List_ID)
IF (Ifpack2_List_ID GREATER -1)
  MESSAGE("-- Looking for Ifpack2 ...                       Found.")
  SET(ALBANY_IFPACK2 TRUE)
ELSE()
  MESSAGE("-- Looking for Ifpack2 ...                   NOT Found.")
  SET(ALBANY_IFPACK2 FALSE)
ENDIF()

# set optional dependency on Teko, for preconditioning
LIST(FIND Trilinos_PACKAGE_LIST Teko Teko_List_ID)
IF (Teko_List_ID GREATER -1)
  MESSAGE("-- Looking for Teko ...                          Found.")
  SET(ALBANY_TEKO TRUE)
ELSE()
  MESSAGE("-- Looking for Teko ...                      NOT Found.")
  SET(ALBANY_TEKO FALSE)
ENDIF()

# set optional dependency on MueLu, for preconditioning
LIST(FIND Trilinos_PACKAGE_LIST MueLu MueLu_List_ID)
IF (MueLu_List_ID GREATER -1)
  MESSAGE("-- Looking for MueLu ...                         Found.")
  SET(ALBANY_MUELU TRUE)
  LIST(FIND MueLu_PACKAGE_LIST Ifpack2 MueLu_Ifpack2_List_ID)
  LIST(FIND MueLu_PACKAGE_LIST Amesos2 MueLu_Amesos2_List_ID)
  IF (MueLu_Ifpack2_List_ID GREATER -1 AND MueLu_Amesos2_List_ID GREATER -1 AND ALBANY_AMESOS2_OPERATIONAL_SOLVER)
    MESSAGE("  -- MueLu tests enabled")
    SET(ALBANY_MUELU_EXAMPLES TRUE)
  ELSE()
    MESSAGE("  -- MueLu tests NOT enabled. Ifpack2 and Amesos2 compiled with an operational solver are required.")
    SET(ALBANY_MUELU_EXAMPLES FALSE)
  ENDIF()
ELSE()
  MESSAGE("-- Looking for MueLu ...                     NOT Found.")
  SET(ALBANY_MUELU FALSE)
  SET(ALBANY_MUELU_EXAMPLES FALSE)
ENDIF()

# set optional dependency on FROSch, for preconditioning
LIST(FIND Trilinos_PACKAGE_LIST ShyLU_DDFROSch FROSch_List_ID)
IF (FROSch_List_ID GREATER -1)
  MESSAGE("-- Looking for FROSch ...                        Found.")
  SET(ALBANY_FROSCH TRUE)
ELSE()
  MESSAGE("-- Looking for FROSch ...                    NOT Found.")
  SET(ALBANY_FROSCH FALSE)
ENDIF()

# Check if Trilinos has ROL
LIST(FIND Trilinos_PACKAGE_LIST ROL ROL_List_ID)
IF (ROL_List_ID GREATER -1)
  MESSAGE("-- Looking for ROL ...                           Found.")
    SET(ALBANY_ROL TRUE)
  ELSE()
    MESSAGE("-- Looking for ROL ...              NOT Found.")
    SET(ALBANY_ROL FALSE)
ENDIF()

# Check if Trilinos has DTK
LIST(FIND Trilinos_PACKAGE_LIST DataTransferKit DataTransferKit_List_ID)
IF (DataTransferKit_List_ID GREATER -1)
  MESSAGE("-- Looking for DTK ...                           Found.")
  SET(ALBANY_DTK TRUE)
ELSE()
  MESSAGE("-- Looking for DTK ...                       NOT Found.")
  SET(ALBANY_DTK FALSE)
ENDIF()

# Set optional dependency on SEACAS, for I/O via stk_io
LIST(FIND Trilinos_PACKAGE_LIST SEACASIoss SEACASIoss_List_ID)
IF (SEACASIoss_List_ID GREATER -1)
  MESSAGE("-- Looking for SEACASIoss ...                    Found.")
  SET(ALBANY_SEACAS TRUE)
  LIST(FIND SEACASIoss_LIBRARIES Iopx Iopx_List_ID)
  IF (Iopx_List_ID GREATER -1)
    MESSAGE("  -- Looking for Iopx ...                        Found.")
    SET(ALBANY_IOPX TRUE)
  ELSE()
    MESSAGE("  -- Looking for Iopx ...                    NOT Found.")
    SET(ALBANY_IOPX FALSE)
  ENDIF()
ELSE()
  MESSAGE("-- Looking for SEACASIoss ...                NOT Found.")
  SET(ALBANY_SEACAS FALSE)
ENDIF()

# Check whether Exodus can access a file in parallel
include(CheckCSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES ${${PROJECT_NAME}_EXTRA_LINK_FLAGS})
set(CMAKE_REQUIRED_INCLUDES  ${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS})
check_c_source_compiles(
  "
  #include <exodusII_par.h>
  #if !NC_HAS_HDF5
  #error Missing hdf5
  #elif !NC_HAS_PNETCDF
  #error Missing pnetcdf
  #endif
  int main()
  {
    return 0;
  }
  "
  PARALLEL_EXODUS_SUPPORTED
  )
  
IF (PARALLEL_EXODUS_SUPPORTED)
  SET(ALBANY_PARALELL_EXODUS TRUE)
  MESSAGE("-- ALBANY_PARALELL_EXODUS set to True")
ELSE()
  SET(ALBANY_PARALELL_EXODUS FALSE)
  MESSAGE("-- ALBANY_PARALELL_EXODUS set to False")
ENDIF()

##########################################
### Check Albany configuration options ###
##########################################

MESSAGE ("\n  ************************************")
MESSAGE ("  *** Albany configuration options ***")
MESSAGE ("  ************************************\n")

# Set optional build of code depending on Epetra.
# Be default, it will be on, so both Tpetra and Epetra capabilities are built.
# Ideally, one should not need Epetra though, since the Tpetra stack in
# Trilinos *should* be a superset of the Epetra one (in terms of capabilities,
# not structures, obviously).

IF (ENABLE_ALBANY_EPETRA)
  MESSAGE("-- ALBANY_EPETRA                           Enabled.")
  SET(ALBANY_EPETRA TRUE)

  # If Epetra build is requested, the ThyraEpetraAdapter package  in Trilinos is required
  LIST(FIND Trilinos_PACKAGE_LIST ThyraEpetraAdapters ThyraEpetraAdapters_ID)
  IF (ThyraEpetraAdapters_ID GREATER -1)
    MESSAGE("  -- Looking for ThyraEpetraAdapters       Found.")
  ELSE()
    MESSAGE("  -- Looking for ThyraEpetraAdapters   NOT Found.")
    MESSAGE (FATAL_ERROR "ENABLE_ALBANY_EPETRA=ON requires ThyraEpetraAdapters in Trilinos.\n")
  ENDIF()
ELSE()
  MESSAGE("-- ALBANY_EPETRA                     NOT Enabled.")
  SET(ALBANY_EPETRA FALSE)
ENDIF()

IF (ENABLE_ALBANY_PYTHON)
  MESSAGE("-- ALBANY_PYTHON                           Enabled.")
  SET(ALBANY_PYTHON TRUE)
ELSE()
  MESSAGE("-- ALBANY_PYTHON                       NOT Enabled.")
  SET(ALBANY_PYTHON FALSE)
ENDIF()

# set optional dependency of mesh on parameters, defaults to Disabled
OPTION(ENABLE_MESH_DEPENDS_ON_PARAMETERS "Flag to turn on dependency of mesh on parameters, e.g for shape optimization" OFF)
IF (ENABLE_MESH_DEPENDS_ON_PARAMETERS)
  MESSAGE("-- MESH_DEPENDS_ON_PARAMETERS              Enabled.")
  SET(ALBANY_MESH_DEPENDS_ON_PARAMETERS TRUE)
ELSE()
  MESSAGE("-- MESH_DEPENDS_ON_PARAMETERS          NOT Enabled.")
  SET(ALBANY_MESH_DEPENDS_ON_PARAMETERS FALSE)
ENDIF()

MESSAGE("\nAlbany physics packages:\n")

# set optional dependency on demoPDEs, defaults to Enabled
OPTION(ENABLE_DEMO_PDES "Flag to turn on demonstration PDEs problems" OFF)
IF (ENABLE_DEMO_PDES)
  MESSAGE("-- DEMO_PDES                               Enabled.")
ELSE()
  MESSAGE("-- DEMO_PDES                           NOT Enabled.")
ENDIF()

# Set optional build of LandIce (Land Ice SciDAC project), defaults to Disabled
OPTION(ENABLE_LANDICE "Flag to turn on LandIce Source code" OFF)
IF (ENABLE_LANDICE)
  MESSAGE("-- LandIce                                 Enabled.")
ELSE()
  MESSAGE("-- LandIce                             NOT Enabled.")
ENDIF()

## Other Albany Template Options here:
MESSAGE("\nAlbany Templated Data Type Options:\n")

# Set FAD data type
SET(ENABLE_FAD_TYPE "DFad" CACHE STRING "Sacado forward mode automatic differentiation data type")
IF(ENABLE_FAD_TYPE STREQUAL "SFad")
  SET(ALBANY_FAD_TYPE_SFAD TRUE)
  SET(ALBANY_SFAD_SIZE 32 CACHE STRING "Number of derivative components chosen at compile-time for AD")
  MESSAGE("-- FAD_TYPE  is SFad, ALBANY_SFAD_SIZE=${ALBANY_SFAD_SIZE}")
  MESSAGE("---> WARNING: problems with elemental DOFs > ${ALBANY_SFAD_SIZE} will fail")
ELSEIF(ENABLE_FAD_TYPE STREQUAL "SLFad")
  SET(ALBANY_FAD_TYPE_SLFAD TRUE)
  SET(ALBANY_SLFAD_SIZE 32 CACHE STRING "Maximum number of derivative components chosen at compile-time for AD")
  MESSAGE("-- FAD_TYPE  is SLFad, ALBANY_SLFAD_SIZE=${ALBANY_SLFAD_SIZE}")
  MESSAGE("---> WARNING: problems with elemental DOFs > ${ALBANY_SLFAD_SIZE} will fail")
ELSEIF(ENABLE_FAD_TYPE STREQUAL "DFad")
  MESSAGE("-- FAD_TYPE  is DFad (default)")
ELSE()
  MESSAGE(FATAL_ERROR
  "\nError: ENABLE_FAD_TYPE = ${ENABLE_FAD_TYPE} is not recognized!
  Options: SFad, SLFad, DFad (default)")
ENDIF()

# Set FAD data type for Tangent
SET(ENABLE_TAN_FAD_TYPE "DFad" CACHE STRING "Sacado forward mode automatic differentiation data type for Tangent")
IF(ENABLE_TAN_FAD_TYPE STREQUAL "SFad")
  SET(ALBANY_TAN_FAD_TYPE_SFAD TRUE)
  SET(ALBANY_TAN_SFAD_SIZE 32 CACHE STRING "Number of derivative components chosen at compile-time for Tangent AD")
  MESSAGE("-- TAN_FAD_TYPE is SFad, ALBANY_TAN_SFAD_SIZE=${ALBANY_TAN_SFAD_SIZE}")
  MESSAGE("---> WARNING: problems with Tangent elemental DOFs > ${ALBANY_TAN_SFAD_SIZE} will fail")
ELSEIF(ENABLE_TAN_FAD_TYPE STREQUAL "SLFad")
  SET(ALBANY_TAN_FAD_TYPE_SLFAD TRUE)
  SET(ALBANY_TAN_SLFAD_SIZE 32 CACHE STRING "Maximum number of derivative components chosen at compile-time for Tangent AD")
  MESSAGE("-- TAN_FAD_TYPE is SLFad, ALBANY_TAN_SLFAD_SIZE=${ALBANY_TAN_SLFAD_SIZE}")
  MESSAGE("---> WARNING: problems with Tangent elemental DOFs > ${ALBANY_TAN_SLFAD_SIZE} will fail")
ELSEIF(ENABLE_TAN_FAD_TYPE STREQUAL "DFad")
  MESSAGE("-- TAN_FAD_TYPE is DFad (default)")
ELSE()
  MESSAGE(FATAL_ERROR
  "\nError: ENABLE_TAN_FAD_TYPE = ${ENABLE_TAN_FAD_TYPE} is not recognized!
  Options: SFad, SLFad, DFad")
ENDIF()

# Set FAD data type for Hessian
SET(ENABLE_HES_VEC_FAD_TYPE "DFad" CACHE STRING "Sacado forward mode automatic differentiation data type for Hessian-vector product")
IF(ENABLE_HES_VEC_FAD_TYPE STREQUAL "SFad")
  SET(ALBANY_HES_VEC_FAD_TYPE_SFAD TRUE)
  SET(ALBANY_HES_VEC_SFAD_SIZE 32 CACHE STRING "Number of derivative components chosen at compile-time for Hessian-vector AD")
  MESSAGE("-- HES_VEC_FAD_TYPE is SFad, ALBANY_HES_VEC_SFAD_SIZE=${ALBANY_HES_VEC_SFAD_SIZE}")
  MESSAGE("---> WARNING: problems with elemental DOFs > ${ALBANY_HES_VEC_SFAD_SIZE} will fail")
ELSEIF(ENABLE_HES_VEC_FAD_TYPE STREQUAL "SLFad")
  SET(ALBANY_HES_VEC_FAD_TYPE_SLFAD TRUE)
  SET(ALBANY_HES_VEC_SLFAD_SIZE 32 CACHE STRING "Maximum number of derivative components chosen at compile-time for Hessian-vector AD")
  MESSAGE("-- HES_VEC_FAD_TYPE is SLFad, ALBANY_HES_VEC_SLFAD_SIZE=${ALBANY_HES_VEC_SLFAD_SIZE}")
  MESSAGE("---> WARNING: problems with elemental DOFs > ${ALBANY_HES_VEC_SLFAD_SIZE} will fail")
ELSEIF(ENABLE_HES_VEC_FAD_TYPE STREQUAL "DFad")
  MESSAGE("-- HES_VEC_FAD_TYPE is DFad (default)")
ELSE()
  MESSAGE(FATAL_ERROR
  "\nError: ENABLE_HES_VEC_FAD_TYPE = ${ENABLE_HES_VEC_FAD_TYPE} is not recognized!
  Options: SFad, SLFad, DFad")
ENDIF()

# Check if FAD data type is the same
IF(ENABLE_FAD_TYPE STREQUAL ENABLE_TAN_FAD_TYPE)
  IF(ALBANY_FAD_TYPE_SFAD AND NOT ALBANY_SFAD_SIZE EQUAL ALBANY_TAN_SFAD_SIZE)
    SET(ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE TRUE)
    MESSAGE("-- SFAD_SIZE is not TAN_SFAD_SIZE")
  ELSEIF(ALBANY_FAD_TYPE_SLFAD AND NOT ALBANY_SLFAD_SIZE EQUAL ALBANY_TAN_SLFAD_SIZE)
    SET(ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE TRUE)
    MESSAGE("-- SLFAD_SIZE is not TAN_SLFAD_SIZE")
  ENDIF()
ELSE()
  SET(ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE TRUE)
  MESSAGE("-- FAD_TYPE is not TAN_FAD_TYPE")
ENDIF()

# Disable the RTC capability if Trilinos is not built with Pamgen
LIST(FIND Trilinos_PACKAGE_LIST Pamgen PAMGEN_List_ID)
  IF (NOT PAMGEN_List_ID GREATER -1)
    set(ALBANY_PAMGEN FALSE)
    MESSAGE("-- Pamgen package not found, disabling run time compiler support.")
  ELSE()
    set(ALBANY_PAMGEN TRUE)
    MESSAGE("-- Pamgen package found, enabling run time compiler support.")
  ENDIF()

## Other Miscellaneous Albany Options here:
MESSAGE("\nMiscellaneous Albany Options:\n")

# Optional CUDA settings
IF(ALBANY_ENABLE_CUDA)
  IF (ENABLE_ALBANY_EPETRA)
    MESSAGE(FATAL_ERROR "\nError: Albany with Epetra is not supported with CUDA KokkosNode!  Please re-configure w/o Epetra (ENABLE_ALBANY_EPETRA=OFF).")
  ENDIF()
  SET(KOKKOS_NDEVICES "--kokkos-map-device-id-by=mpi_rank")
  IF (ENABLE_CUDA_ERROR_CHECK)
    SET(ALBANY_CUDA_ERROR_CHECK ON)
    MESSAGE("-- CUDA Error Check        Enabled")
  ELSE()
    MESSAGE("-- CUDA Error Check    NOT Enabled.")
  ENDIF()
  IF (ENABLE_CUDA_NVTX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lnvToolsExt")
    SET(ALBANY_CUDA_NVTX ON)
    MESSAGE("-- CUDA NVTX               Enabled")
  ELSE()
    MESSAGE("-- CUDA NVTX           NOT Enabled.")
  ENDIF()
ENDIF()

# Add option to check for hardware floating point exceptions using SSE, defaults to disabled
OPTION(ENABLE_CHECK_FPE "Flag to turn on floating point exception (NAN) checking" OFF)

# Add option to check for hardware floating point exceptions using SSE, defaults to disabled
OPTION(ENABLE_CHECK_FPE "Flag to turn on floating point exception (NAN) checking" OFF)
IF (ENABLE_CHECK_FPE)
  MESSAGE("-- FPE checking              Enabled")
  SET(ALBANY_CHECK_FPE TRUE)
ELSE()
  MESSAGE("-- FPE checking          NOT Enabled.")
  SET(ALBANY_CHECK_FPE FALSE)
ENDIF()

OPTION(ENABLE_STRONG_FPE_CHECK "Flag to turn on strong checking for floating point exceptions" OFF)
IF (ENABLE_STRONG_FPE_CHECK)
  MESSAGE("-- Strong FPE checking       Enabled")
  SET(ALBANY_STRONG_FPE_CHECK TRUE)
ELSE()
  MESSAGE("-- Strong FPE checking   NOT Enabled.")
  SET(ALBANY_STRONG_FPE_CHECK FALSE)
ENDIF()

# Add option to set FTZ and DAZ
OPTION(ENABLE_FLUSH_DENORMALS "Flag to turn on FTZ and DAZ" OFF)
IF (ENABLE_FLUSH_DENORMALS)
  MESSAGE("-- FTZ/DAZ flags             Enabled")
  SET(ALBANY_FLUSH_DENORMALS TRUE)
ELSE()
  MESSAGE("-- FTZ/DAZ flags         NOT Enabled.")
  SET(ALBANY_FLUSH_DENORMALS FALSE)
ENDIF()

#amb Default to on until the merge.
OPTION(ENABLE_KOKKOS_UNDER_DEVELOPMENT "Flag to turn on Kokkos experimental code in Albany" ON)
IF (ENABLE_KOKKOS_UNDER_DEVELOPMENT)
  MESSAGE("-- Kokkos development        Enabled")
  SET(ALBANY_KOKKOS_UNDER_DEVELOPMENT TRUE)
ELSE()
  MESSAGE("-- Kokkos development    NOT Enabled.")
  SET(ALBANY_KOKKOS_UNDER_DEVELOPMENT FALSE)
ENDIF()

OPTION(ENABLE_ALBANY_VERBOSE "Flag to turn on verbose output" OFF)
IF (ENABLE_ALBANY_VERBOSE)
  MESSAGE("-- Verbose printing          Enabled")
  SET(ALBANY_VERBOSE TRUE)
ELSE()
  MESSAGE("-- Verbose printing      NOT Enabled")
  SET(ALBANY_VERBOSE FALSE)
ENDIF()

# Enhanced debugging option
OPTION(ENABLE_DEBUGGING "Flag to turn on enhanced debugging info" OFF)
IF (ENABLE_DEBUGGING)
  MESSAGE("-- Enhanced debugging        Enabled")
  SET(ALBANY_DEBUG TRUE)
ELSE()
  MESSAGE("-- Enhanced debugging    NOT Enabled.")
  SET(ALBANY_DEBUG FALSE)
ENDIF()

# Set which memory analysis system routines are available.
option(ENABLE_MALLINFO "Use mallinfo() in Albany::printMemoryAnalysis()" OFF)
option(ENABLE_GETRUSAGE "Use getrusage() in Albany::printMemoryAnalysis()" OFF)
option(ENABLE_KERNELGETMEMORYSIZE "Use Kernel_GetMemorySize() in Albany::printMemoryAnalysis()" OFF)
if (ENABLE_MALLINFO)
  SET(ALBANY_HAVE_MALLINFO TRUE)
  message("-- Memory: mallinfo()        Enabled.")
endif()
if (ENABLE_GETRUSAGE)
  SET(ALBANY_HAVE_GETRUSAGE TRUE)
  message("-- Memory: getrusage()       Enabled.")
endif()
if (ENABLE_KERNELGETMEMORYSIZE)
  SET(ALBANY_HAVE_KERNELGETMEMORYSIZE ON)
  message("-- Memory: Kernel_GetMemorySize() Enabled.")
endif()

# Mesh database tools.
OPTION(ENABLE_MESHDB_TOOLS "Flag to turn on mesh database tools" OFF)
IF (ENABLE_MESHDB_TOOLS)
  MESSAGE("-- Mesh database tools       Enabled.")
  SET(ALBANY_MESHDB_TOOLS TRUE)
ENDIF()

# Omega_h
OPTION(ENABLE_OMEGAH "Flag to turn on the Omega_h mesh interface" OFF)
IF (ENABLE_OMEGAH)
  MESSAGE("-- Omega_h mesh interface    Enabled.")
  SET(ALBANY_OMEGAH TRUE)
  message (STATUS "Looking for Omega_h installation ...")
  find_package(Omega_h CONFIG
               HINTS ${Albany_BINARY_DIR}/tpls/omegah/install)
  if (NOT Omega_h_FOUND)
    message (STATUS "Looking for Omega_h installation ... NOT Found.")
    message ("  +--------------------------------------------------------+")
    message ("  |      Begin local installation of Omega_h dependency    |")
    message ("  +--------------------------------------------------------+")
    # The first external project will be built at *configure stage*
    # Configure
    execute_process(
      COMMAND ${CMAKE_COMMAND}
              -B ${Albany_BINARY_DIR}/tpls/omegah/tmp -Wno-dev
              -S ${CMAKE_SOURCE_DIR}/cmake/omegah
              -D Albany_BINARY_DIR=${Albany_BINARY_DIR}
              -D Kokkos_INSTALL_DIR=${ALBANY_TRILINOS_DIR}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      OUTPUT_VARIABLE CONFIG_OMEGAH_OUT
      ERROR_VARIABLE CONFIG_OMEGAH_OUT
      RESULT_VARIABLE CONFIG_OMEGAH_RES
      ECHO_OUTPUT_VARIABLE
    )
    if (NOT CONFIG_OMEGAH_RES EQUAL 0)
      message ("Could not configure omegah")
      message (" output:")
      message ("${CONFIG_OMEGAH_OUT}")
      message (FATAL_ERROR "Die")
    endif()
    # Build/install
    execute_process(
      COMMAND ${CMAKE_COMMAND} --build
              ${Albany_BINARY_DIR}/tpls/omegah/tmp
              --parallel 8
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      OUTPUT_VARIABLE BUILD_OMEGAH_OUT
      ERROR_VARIABLE BUILD_OMEGAH_OUT
      RESULT_VARIABLE BUILD_OMEGAH_RES
    )
    if (NOT BUILD_OMEGAH_RES EQUAL 0)
      message ("Could not build omegah")
      message (" output:")
      message ("${BUILD_OMEGAH_OUT}")
      message (FATAL_ERROR "Die")
    endif()
    message ("  +--------------------------------------------------------+")
    message ("  |       End local installation of Omega_h dependency     |")
    message ("  +--------------------------------------------------------+")
    set (Omega_h_DIR ${Albany_BINARY_DIR}/tpls/omegah/install)
    find_package (Omega_h REQUIRED
                  HINTS ${Albany_BINARY_DIR}/tpls/omegah/install)
  else()
    message (STATUS "Looking for Omega_h installation ... Found.")
    message ("  -- Omega_h prefix: ${Omega_h_DIR}")
    message ("  -- Omega_h version: ${Omega_h_VERSION}")
  endif()
ENDIF()

# add a target to generate API documentation with Doxygen
IF( NOT DISABLE_DOXYGEN )
  find_package(Doxygen)
  if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/albanyMainPage.doc ${CMAKE_CURRENT_BINARY_DIR}/albanyMainPage.doc)
  add_custom_target(doc
  ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating API documentation with Doxygen" VERBATIM
  )
  endif(DOXYGEN_FOUND)
ENDIF()

# Enables experimental features/tests. This may cause compilation errors and/or
# tests failures, as it includes code that is currently developed.

OPTION (ENABLE_EXPERIMENTAL "Enables experimental features/tests." OFF)

# Disable ability to write to output mesh. On slow filesystems, this can give faster
# execution time, if one is only interested in regression test against baseline
# response/sensitivity values.
option (ALBANY_DISABLE_OUTPUT_MESH "Disable writing of output mesh files" OFF)

########################## End of CMake options to Albany #########################################

# Parse source code folder
add_subdirectory(src)

IF(ALBANY_MPI_EXEC)
  SET(MPIEX  ${ALBANY_MPI_EXEC})
ELSE()
  SET(MPIEX  ${Trilinos_MPI_EXEC})
ENDIF()
IF(ALBANY_MPI_EXEC_NUMPROCS_FLAG)
  SET(MPINPF  ${ALBANY_MPI_EXEC_NUMPROCS_FLAG})
ELSE()
  SET(MPINPF ${Trilinos_MPI_EXEC_NUMPROCS_FLAG})
ENDIF()
IF(ALBANY_MPI_EXEC_MAX_NUMPROCS)
  SET(MPIMNP ${ALBANY_MPI_EXEC_MAX_NUMPROCS})
ELSE()
  SET(MPIMNP ${Trilinos_MPI_EXEC_MAX_NUMPROCS})
ENDIF()
IF(ALBANY_MPI_EXEC_LEADING_OPTIONS)
  SET(MPIPRE ${ALBANY_MPI_EXEC_LEADING_OPTIONS})
  SEPARATE_ARGUMENTS(MPIPRE)
ENDIF()
IF(ALBANY_MPI_EXEC_TRAILING_OPTIONS)
  SET(MPIPOST ${ALBANY_MPI_EXEC_TRAILING_OPTIONS})
  SEPARATE_ARGUMENTS(MPIPOST)
ENDIF()
SET(SERIAL_CALL ${MPIEX} ${MPIPRE} ${MPINPF} 1 ${MPIPOST})
SET(PARALLEL_CALL ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST})
STRING(REPLACE ";" " " SERIAL_CALL_TEXT "${SERIAL_CALL}")
STRING(REPLACE ";" " " PARALLEL_CALL_TEXT "${PARALLEL_CALL}")
MESSAGE("-- CTest serial call: ${SERIAL_CALL_TEXT}")
MESSAGE("-- CTest parallel call: ${PARALLEL_CALL_TEXT}")

#Add option to build libraries but not executables (for LandIce uner MPAS). Disables all tests
option(ALBANY_LIBRARIES_ONLY "Flag to only build libraries and not link executables" OFF)
#Add option to disable all tests
option(DISABLE_ALBANY_TESTS "Flag to disable tests" OFF)
if (ALBANY_LIBRARIES_ONLY)
  message("\nALBANY_LIBRARIES_ONLY has been set: turning off all tests.")
elseif (DISABLE_ALBANY_TESTS)
  message("\nDISABLE_ALBANY_TESTS has been set: turning off all tests.")
else()
  # Add Albany test -- in the "tests" directory
  add_subdirectory(tests)
endif()

IF (ALBANY_PYTHON)
  IF (ALBANY_ROL)
    add_subdirectory( pyAlbany )
  ELSE()
    MESSAGE(FATAL_ERROR "PyAlbany requires a Trilinos installation that enables ROL.")
  ENDIF()
ENDIF()

# Make a clean build target for convenience

ADD_CUSTOM_TARGET( cmake_clean
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/Makefile
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/cmake_install.cmake
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake
)

MESSAGE("\nAnd CMake says...:")
