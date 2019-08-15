##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//


# CMAKE File for Albany building against an installed Trilinos

cmake_minimum_required(VERSION 2.7)
include(CMakeDependentOption)

OPTION (USE_NEW_POLICY_CMP0060 "Flag to turn on new behavior for policy CMP0060" OFF)
IF (USE_NEW_POLICY_CMP0060)
  CMAKE_POLICY(SET CMP0060 NEW)
ENDIF()

if (NOT DEFINED SET_COMPILERS_AUTOMATICALLY)
  set(SET_COMPILERS_AUTOMATICALLY TRUE)
endif()

if (NOT SET_COMPILERS_AUTOMATICALLY)
  message("Experimental configuration.")
  project(Albany CXX C Fortran)
endif ()

# Advertise that we support shared libs
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

# Get Trilinos as one entity
IF (DEFINED ALBANY_TRILINOS_DIR)
  SET(CMAKE_PREFIX_PATH ${ALBANY_TRILINOS_DIR} ${CMAKE_PREFIX_PATH})
ENDIF()
FIND_PACKAGE(Trilinos REQUIRED)

OPTION (ALBANY_SUPPRESS_TRILINOS_WARNINGS "Whether or not Trilinos headers should be treated as 'system' headers (hence, without issuing warnings)" OFF)

# Trilinos_BIN_DIRS probably should be defined in the Trilinos config. Until it is, set it here.
# This is needed to find SEACAS tools used during testing (epu, etc).

SET(Trilinos_BIN_DIRS "${Trilinos_DIR}/../../../bin")
SET(Trilinos_LIB_DIRS "${Trilinos_DIR}/../../../lib")

MESSAGE("\nFound Trilinos!  Here are the details: ")
MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
#MESSAGE("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
#MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
MESSAGE("   Trilinos_BIN_DIRS = ${Trilinos_BIN_DIRS}")
#MESSAGE("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
#MESSAGE("   Trilinos_LIBRARY_DIRS = ${Trilinos_LIBRARY_DIRS}")
MESSAGE("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
#MESSAGE("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
#MESSAGE("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
MESSAGE("   Trilinos_TPL_LIBRARY_DIRS = ${Trilinos_TPL_LIBRARY_DIRS}")
MESSAGE("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
MESSAGE("   Trilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
MESSAGE("End of Trilinos details\n")

# Compress Trilinos lists that are used
#${Trilinos_INCLUDE_DIRS}
#${Trilinos_TPL_INCLUDE_DIRS}
#${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} ${Trilinos_EXTRA_LD_FLAGS}

LIST(REVERSE Trilinos_INCLUDE_DIRS)
LIST(REMOVE_DUPLICATES Trilinos_INCLUDE_DIRS)
LIST(REVERSE Trilinos_INCLUDE_DIRS)

LIST(REVERSE Trilinos_TPL_INCLUDE_DIRS)
LIST(REMOVE_DUPLICATES Trilinos_TPL_INCLUDE_DIRS)
LIST(REVERSE Trilinos_TPL_INCLUDE_DIRS)

LIST(REVERSE Trilinos_LIBRARIES)
LIST(REMOVE_DUPLICATES Trilinos_LIBRARIES)
LIST(REVERSE Trilinos_LIBRARIES)

LIST(REVERSE Trilinos_TPL_LIBRARIES)
LIST(REMOVE_DUPLICATES Trilinos_TPL_LIBRARIES)
LIST(REVERSE Trilinos_TPL_LIBRARIES)

LIST(FIND Trilinos_PACKAGE_LIST STKClassic STKClassic_List_ID)
IF (STKClassic_List_ID GREATER -1)
  MESSAGE(FATAL_ERROR "STKClassic is no longer used and conflicts with STKIo and STKMesh.")
ENDIF()

# Optional Installation helpers
# Note that some of this functionality depends on CMAKE > 2.8.8
SET (INSTALL_ALBANY FALSE CACHE BOOL "Whether Albany should be installed")
IF (INSTALL_ALBANY)
  IF (${CMAKE_VERSION} VERSION_GREATER 2.8.8 OR ${CMAKE_VERSION} VERSION_EQUAL 2.8.8)
    MESSAGE("")
    MESSAGE("Installation files for Albany will be created")
    MESSAGE("-- install location = ${CMAKE_INSTALL_PREFIX}")
    MESSAGE("")
    set(BINARY_INSTALL_DIR bin)
    set(INCLUDE_INSTALL_DIR include)
    set(LIB_INSTALL_DIR lib)
    include(CMakePackageConfigHelpers)
  ELSE()
    MESSAGE("")
    MESSAGE("Installation has been requested, but cannot continue")
    MESSAGE("-- Please upgrade CMAKE to version >= 2.8.8")
    MESSAGE("")
  ENDIF()
ENDIF()

option (ENABLE_CROSS_COMPILE "Flag to turn on cross-compiling tools" OFF)
if (ENABLE_CROSS_COMPILE)
  set (Albany_CROSS_COMPILE ON)
  message ("Building with cross-compiling tools.")
else ()
  set (Albany_CROSS_COMPILE OFF)
endif ()

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


# End of setup and error checking
#  NOTE: PROJECT command checks for compilers, so this statement
#        is moved AFTER setting CMAKE_CXX_COMPILER from Trilinos

if (SET_COMPILERS_AUTOMATICALLY)
  PROJECT(Albany)
endif()

# AGS: Adding these lines so Ctest can be run to submit to cdash dashboard
#   Uses CTestConfig.cmake file for Cdash info.
ENABLE_TESTING()
INCLUDE(CTest)

set(ALBANY_ENABLE_FORTRAN ON CACHE BOOL "enable fortran" )

IF (CMAKE_Fortran_COMPILER AND ALBANY_ENABLE_FORTRAN) # Enable Fortran if it is enabled in Trilinos.
  ENABLE_LANGUAGE(Fortran)
ENDIF()

# Make Albany inherit Trilinos compiler flags. The most important are the
# presence or absence of -NDEBUG (which can affect linking for complicated
# reasons) and -std=c++11.
SET(CMAKE_CXX_FLAGS "${Trilinos_CXX_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}")
# GAH - temporarily set Albany flags for clang debugging
#SET(CMAKE_CXX_FLAGS "-ggdb -std=c++11 -O2 -march=native -DNDEBUG")

if (SET_COMPILERS_AUTOMATICALLY)
  SET(CMAKE_AR ${Trilinos_AR})
  SET(CMAKE_LINKER ${Trilinos_LINKER})
endif()

MESSAGE("-- Using ${Trilinos_CXX_COMPILER_ID} Compiler.")

# No longer needed but keep for now as this shows how to silence specific warnings on Intel compiler
#IF(${Trilinos_CXX_COMPILER_ID} MATCHES "Intel")
#  SET(USING_INTEL TRUE)
# On Intel, turn of interprocedural optimization at linking to keep linker from hanging
# This could possibly be removed with further study (GAH)
#  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-ip")
#   SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd2536,2196,858,488,161,47")
#ENDIF()

OPTION(ENABLE_ALBANY_EPETRA "Flag to turn on building of code depending on Epetra" ON)

## Check for some required Trilinos package dependencies (Piro, Phalanx) ##
MESSAGE("\nChecking Trilinos build for required and optional packages")

###############################
### Check REQUIRED packages ###
###############################

MESSAGE ("\n  **********************************")
MESSAGE ("  *** Required Trilinos Packages ***")
MESSAGE ("  **********************************\n")
# Piro
LIST(FIND Trilinos_PACKAGE_LIST Piro Piro_List_ID)
IF (Piro_List_ID GREATER -1)
  MESSAGE("-- Looking for Piro (required) in Trilinos ...                Found!")
ELSE()
  MESSAGE("-- Looking for Piro (required) in Trilinos ...                NOT Found!")
  MESSAGE(FATAL_ERROR "Albany REQUIRES Piro. Please, configure Trilinos with guidance from trilinos-cmake script in doc directory.")
ENDIF()

# Phalanx
LIST(FIND Trilinos_PACKAGE_LIST Phalanx Phalanx_List_ID)
IF (Piro_List_ID GREATER -1)
  MESSAGE("-- Looking for Phalanx (required) in Trilinos ...             Found!")
ELSE()
  MESSAGE("-- Looking for Phalanx (required) in Trilinos ...             NOT Found!")
  MESSAGE(FATAL_ERROR "Albany REQUIRES Phalanx. Please, configure Trilinos with guidance from trilinos-cmake script in doc directory.")
ENDIF()

# NOX
LIST(FIND Trilinos_PACKAGE_LIST NOX NOX_List_ID)
IF (Piro_List_ID GREATER -1)
  MESSAGE("-- Looking for NOX (required) in Trilinos ...                 Found!")
ELSE()
  MESSAGE("-- Looking for NOX (required) in Trilinos ...                 NOT Found!")
  MESSAGE(FATAL_ERROR "Albany REQUIRES NOX. Please, configure Trilinos with guidance from trilinos-cmake script in doc directory.")
ENDIF()

# ThyraTpetraAdapters
LIST(FIND Trilinos_PACKAGE_LIST ThyraTpetraAdapters ThyraTpetraAdapters_List_ID)
IF (Piro_List_ID GREATER -1)
  MESSAGE("-- Looking for ThyraTpetraAdapters (required) in Trilinos ... Found!")
ELSE()
  MESSAGE("-- Looking for ThyraTpetraAdapters (required) in Trilinos ... NOT Found!")
  MESSAGE(FATAL_ERROR "Albany REQUIRES ThyraTpetraAdapters. Please, configure Trilinos with guidance from trilinos-cmake script in doc directory.")
ENDIF()

# Intrepid2
LIST(FIND Trilinos_PACKAGE_LIST Intrepid2 Intrepid2_List_ID)
IF (Piro_List_ID GREATER -1)
  MESSAGE("-- Looking for Intrepid2 (required) in Trilinos ...           Found!")
ELSE()
  MESSAGE("-- Looking for Intrepid2 (required) in Trilinos ...           NOT Found!")
  MESSAGE(FATAL_ERROR "Albany REQUIRES Intrepid2. Please, configure Trilinos with guidance from trilinos-cmake script in doc directory.")
ENDIF()

# MiniTensor
LIST(FIND Trilinos_PACKAGE_LIST MiniTensor MiniTensor_List_ID)
IF (Piro_List_ID GREATER -1)
  MESSAGE("-- Looking for MiniTensor (required) in Trilinos ...          Found!")
ELSE()
  MESSAGE("-- Looking for MiniTensor (required) in Trilinos ...          NOT Found!")
  MESSAGE(FATAL_ERROR "Albany REQUIRES MiniTensor. Please, configure Trilinos with guidance from trilinos-cmake script in doc directory.")
ENDIF()

###############################
### Check OPTIONAL packages ###
###############################

MESSAGE ("\n  **********************************")
MESSAGE ("  *** Optional Trilinos Packages ***")
MESSAGE ("  **********************************\n")
### Check for some OPTIONAL trilinos packages, which set DEFINES in Alany build ###

# MPI check -- defaults to TRUE
# MESSAGE("-- Checking if MPI is enabled in Trilinos:")

LIST(FIND Trilinos_TPL_LIST MPI MPI_List_ID)
IF (MPI_List_ID GREATER -1)
  MESSAGE("-- Looking for MPI in Trilinos:      Found")
  SET(ALBANY_MPI TRUE)
ELSE()
  MESSAGE("-- Looking for MPI in Trilinos:      NOT found, MPI NOT ENABLED!")
  SET(ALBANY_MPI FALSE)
ENDIF()

# set optional dependency on stk
LIST(FIND Trilinos_PACKAGE_LIST STK have_stk_List_ID)
IF (have_stk_List_ID GREATER -1)
  MESSAGE("-- Looking for STK:                  Found")
  SET(ALBANY_STK TRUE)

  # set optional dependency on SEACAS, for I/O via stk_io
  # MESSAGE("-- Looking for SEACAS:")
  LIST(FIND Trilinos_PACKAGE_LIST SEACASIoss SEACASIoss_List_ID)
  IF (SEACASIoss_List_ID GREATER -1)
    MESSAGE("  -- Looking for SEACASIoss:         Found")
    SET(ALBANY_SEACAS TRUE)
    LIST(FIND SEACASIoss_LIBRARIES Iopx Iopx_List_ID)
    IF (Iopx_List_ID GREATER -1)
      MESSAGE("  -- Looking for Iopx:               Found")
      SET(ALBANY_IOPX TRUE)
    ELSE()
      MESSAGE("  -- Looking for Iopx:               NOT found.")
      SET(ALBANY_IOPX FALSE)
    ENDIF()
  ELSE()
    MESSAGE("  -- Looking for SEACASIoss:         NOT found.")
    SET(ALBANY_SEACAS FALSE)
  ENDIF()

  # set optional dependency on stk_percept, for uniform adaptivity
  LIST(FIND Trilinos_LIBRARIES stk_percept stk_percept_List_ID)
  IF (stk_percept_List_ID GREATER -1)
    MESSAGE("  -- Looking for stk_percept:        Found")
    SET(ALBANY_STK_PERCEPT TRUE)
  ELSE()
    MESSAGE("  -- Looking for stk_percept:        NOT found.")
    SET(ALBANY_STK_PERCEPT FALSE)
  ENDIF()

ELSE()
  MESSAGE("-- Looking for STK:                  NOT found.")
  SET(ALBANY_STK FALSE)
ENDIF()

# set optional dependency on Panzer's ExprEval subpackage
LIST (FIND Trilinos_PACKAGE_LIST PanzerExprEval PanzerExprEval_List_ID)
IF (PanzerExprEval_List_ID GREATER -1)
  MESSAGE("-- Looking for PanzerExprEval:       Found")
  SET(ALBANY_PANZER_EXPR_EVAL TRUE)
ELSE ()
  MESSAGE("-- Looking for PanzerExprEval:       NOT found.")
  SET(ALBANY_PANZER_EXPR_EVAL FALSE)
ENDIF()

# set optional dependency on Zoltan, for partitioning
# MESSAGE("-- Looking for Zoltan:")
LIST(FIND Trilinos_PACKAGE_LIST Zoltan Zoltan_List_ID)
IF (Zoltan_List_ID GREATER -1)
  SET(ALBANY_ZOLTAN TRUE)
  IF(ALBANY_STK)
    SET(ALBANY_STK_REBALANCE TRUE)
    MESSAGE("-- Looking for Zoltan:               Found.")
  ELSE()
    SET(ALBANY_STK_REBALANCE FALSE)
    MESSAGE("-- Looking for Zoltan:               Found.")
  ENDIF()
ELSE()
  MESSAGE("-- Looking for Zoltan:               NOT found.")
  SET(ALBANY_ZOLTAN FALSE)
  SET(ALBANY_STK_REBALANCE FALSE)
ENDIF()

# set optional dependency on Belos's Epetra subpackage
IF (EXISTS "${Trilinos_INCLUDE_DIRS}/BelosEpetraAdapter.hpp")
  MESSAGE("-- Looking for BelosEpetra:          Found.")
  SET(ALBANY_BELOS_EPETRA TRUE)
ELSE ()
  MESSAGE("-- Looking for BelosEpetra:          NOT found.")
  SET(ALBANY_BELOS_EPETRA FALSE)
ENDIF()

# set optional dependency on Amesos2, for solving and preconditioning
LIST(FIND Trilinos_PACKAGE_LIST Amesos2 Amesos2_List_ID)
IF (Amesos2_List_ID GREATER -1)
  MESSAGE("-- Looking for Amesos2:              Found.")
  SET(ALBANY_AMESOS2 TRUE)
  # We can't rely on Amesos2_TPL_LIST any longer; KLU2 and Basker are not
  # TPLs. These will soon be enabled by default in Amesos2. So we'll just
  # assume we have a coarse direct solver.
  SET(ALBANY_AMESOS2_OPERATIONAL_SOLVER TRUE)
ELSE()
  MESSAGE("-- Looking for Amesos2:              NOT found.")
  SET(ALBANY_AMESOS2 FALSE)
  SET(ALBANY_AMESOS2_OPERATIONAL_SOLVER FALSE)
ENDIF()

# set optional dependency on Ifpack2, for preconditioning
LIST(FIND Trilinos_PACKAGE_LIST Ifpack2 Ifpack2_List_ID)
IF (Ifpack2_List_ID GREATER -1)
  MESSAGE("-- Looking for Ifpack2:              Found.")
  SET(ALBANY_IFPACK2 TRUE)
ELSE()
  MESSAGE("-- Looking for Ifpack2:              NOT found.")
  SET(ALBANY_IFPACK2 FALSE)
ENDIF()

# set optional dependency on Teko, for preconditioning
LIST(FIND Trilinos_PACKAGE_LIST Teko Teko_List_ID)
IF (Teko_List_ID GREATER -1)
  MESSAGE("-- Looking for Teko:                 Found.")
  SET(ALBANY_TEKO TRUE)
ELSE()
  MESSAGE("-- Looking for Teko:                 NOT found.")
  SET(ALBANY_TEKO FALSE)
ENDIF()

# set optional dependency on MueLu, for preconditioning
LIST(FIND Trilinos_PACKAGE_LIST MueLu MueLu_List_ID)
IF (MueLu_List_ID GREATER -1)
  MESSAGE("-- Looking for MueLu:                Found.")
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
  MESSAGE("-- Looking for MueLu:                NOT found.")
  SET(ALBANY_MUELU FALSE)
  SET(ALBANY_MUELU_EXAMPLES FALSE)
ENDIF()

# set optional dependency on FROSch, for preconditioning
LIST(FIND Trilinos_PACKAGE_LIST ShyLU_DDFROSch FROSch_List_ID)
IF (FROSch_List_ID GREATER -1)
  MESSAGE("-- Looking for FROSch:               Found.")
  SET(ALBANY_FROSCH TRUE)
ELSE()
  MESSAGE("-- Looking for FROSch:               NOT found.")
  SET(ALBANY_FROSCH FALSE)
ENDIF()

LIST(FIND Trilinos_PACKAGE_LIST Rythmos Rythmos_List_ID)
IF (Rythmos_List_ID GREATER -1)
  MESSAGE("-- Looking for Rythmos:              Found.")
  SET(ALBANY_RYTHMOS TRUE)
ELSE()
  MESSAGE("-- Looking for Rythmos:              NOT found.")
  SET(ALBANY_RYTHMOS FALSE)
ENDIF()

# Check if Trilinos has ROL
LIST(FIND Trilinos_PACKAGE_LIST ROL ROL_List_ID)
IF (ROL_List_ID GREATER -1)
  MESSAGE("-- Looking for ROL in Trilinos:      Found")
    SET(ALBANY_ROL TRUE)
  ELSE()
    MESSAGE("-- Looking for ROL in Trilinos:    NOT found.")
    SET(ALBANY_ROL FALSE)
ENDIF()

# Check if Trilinos has DTK
LIST(FIND Trilinos_PACKAGE_LIST DataTransferKit DataTransferKit_List_ID)
IF (DataTransferKit_List_ID GREATER -1)
  MESSAGE("-- Looking for DTK in Trilinos:      Found")
  SET(ALBANY_DTK TRUE)
ELSE()
  MESSAGE("-- Looking for DTK in Trilinos:      NOT found.")
  SET(ALBANY_DTK FALSE)
ENDIF()

# Check if Trilinos has Tempus
LIST(FIND Trilinos_PACKAGE_LIST Tempus Tempus_List_ID)
IF (Tempus_List_ID GREATER -1)
  MESSAGE("-- Looking for Tempus in Trilinos:   Found")
  SET(ALBANY_TEMPUS TRUE)
ELSE()
  MESSAGE("-- Looking for Tempus in Trilinos:   NOT found.")
  IF (ENABLE_LCM)
    MESSAGE(FATAL_ERROR
        "LCM enabled but Tempus not installed from Trilinos (LCM needs Tempus)!  Please recompile Trilinos with Tempus enabled.")
  ENDIF()
  SET(ALBANY_TEMPUS FALSE)
ENDIF()

# Set optional build of code depending on Epetra.
# Be default, it will be on, so both Tpetra and Epetra capabilities are built.
# Ideally, one should not need Epetra though, since the Tpetra stack in
# Trilinos *should* be a superset of the Epetra one (in terms of capabilities,
# not structures, obviously).

IF (ENABLE_ALBANY_EPETRA)
  MESSAGE("\n-- ALBANY_EPETRA is Enabled. Checking for additional Trilinos packages")
  SET(ALBANY_EPETRA TRUE)

  LIST(FIND Trilinos_PACKAGE_LIST ThyraEpetraAdapters ThyraEpetraAdapters_ID)
  IF (ThyraEpetraAdapters_ID GREATER -1)
    MESSAGE("-- Looking for ThyraEpetraAdapters:  Found.")
  ELSE()
    MESSAGE("-- Looking for ThyraEpetraAdapters:  NOT found.")
    MESSAGE (FATAL_ERROR "ENABLE_ALBANY_EPETRA=ON requires ThyraEpetraAdapters in Trilinos.\n")
  ENDIF()

  # DAKOTA uses Epetra currently
  LIST(FIND Trilinos_PACKAGE_LIST TriKota TriKota_List_ID)
  IF (TriKota_List_ID GREATER -1)
    MESSAGE("-- Looking for Dakota via TriKota:   Found.")
    SET(ALBANY_DAKOTA TRUE)
  ELSE()
    MESSAGE("-- Looking for Dakota via TriKota:   NOT found.")
    SET(ALBANY_DAKOTA FALSE)
  ENDIF()

ELSE()
  MESSAGE("-- ALBANY_EPETRA     is NOT Enabled.")
  SET(ALBANY_EPETRA FALSE)
  SET(ALBANY_DAKOTA FALSE)
ENDIF()

#################################################
### Check for optional physics sets in Albany ###
#################################################

MESSAGE("\nChecking which Albany physics sets are enabled:")

# set optional dependency on demoPDEs, defaults to Enabled
OPTION(ENABLE_DEMO_PDES "Flag to turn on demonstration PDEs problems" ON)
IF (ENABLE_DEMO_PDES)
  MESSAGE("-- DEMO_PDES                       is Enabled")
  SET(ALBANY_DEMO_PDES TRUE)
ELSE()
  MESSAGE("-- DEMO_PDES                       is NOT Enabled.")
  SET(ALBANY_DEMO_PDES FALSE)
ENDIF()

# set optional dependency on ATO, defaults to Disabled
OPTION(ENABLE_LCM "Flag to turn on LCM Mechanics source code" OFF)
OPTION(ENABLE_ATO "Flag to turn on ATO source code" OFF)
IF (ENABLE_ATO)
  MESSAGE("-- ATO                             is Enabled")
  SET(ALBANY_ATO TRUE)
  IF (ENABLE_ISOLIB)
    SET(ATO_USES_ISOLIB TRUE)
  ENDIF()
  IF (ENABLE_DOTK)
    SET(ATO_USES_DOTK TRUE)
  ENDIF()
  IF (ENABLE_NLopt)
    SET(ATO_USES_NLOPT TRUE)
  ENDIF()
  IF (ENABLE_Cogent)
    SET(ATO_USES_COGENT TRUE)
  ENDIF()
ELSE()
  MESSAGE("-- ATO                             is NOT Enabled.")
  SET(ALBANY_ATO FALSE)
ENDIF()

# set optional dependency on LCM, defaults to Disabled
OPTION(ENABLE_LCM_TEST_EXES "Flag to turn on LCM Mechanics test executables" ON)
OPTION(DISABLE_LCM_EXODIFF_SENSITIVE_TESTS "Flag to turn off LCM exodiff sensitive tests" OFF)
SET(LCM_TEST_EXES FALSE)
IF (ENABLE_LCM OR ENABLE_ATO)
  IF (NOT ENABLE_LCM)
    MESSAGE("-- LCM                             is Enabled because ATO requires LCM.")
  ELSE ()
    MESSAGE("-- LCM                             is Enabled.")
  ENDIF ()
  SET(ALBANY_LCM TRUE)
  IF (ENABLE_LCM AND NOT ALBANY_ROL)
    MESSAGE(FATAL_ERROR
        "LCM enabled but ROL not installed from Trilinos (LCM needs ROL)")
  ENDIF()
  IF (ENABLE_LCM_TEST_EXES AND ALBANY_STK) # Need STK for the EXES
    SET(LCM_TEST_EXES TRUE)
  ENDIF()
  IF (DISABLE_LCM_EXODIFF_SENSITIVE_TESTS)
    MESSAGE("  -- LCM exodiff sensitive tests are DISABLED!")
    SET(DISABLE_LCM_EXODIFF_SENSITIVE_TESTS TRUE)
  ENDIF()
ELSE()
  MESSAGE("-- LCM                             is NOT Enabled.")
  SET(ALBANY_LCM FALSE)
ENDIF()

# set optional dependency on MPMD, defaults to Disabled
OPTION(ENABLE_MPMD "Flag to turn on MPMD source code" OFF)
IF (ENABLE_MPMD)
  MESSAGE("-- MPMP executable                 is Enabled.")
  SET(ALBANY_MPMD TRUE)
ELSE ()
  MESSAGE("-- MPMP executable                 is NOT Enabled.")
  SET(ALBANY_MPMD FALSE)
ENDIF()

# Set optional build of LandIce (Land Ice SciDAC project), defaults to Disabled
OPTION(ENABLE_LANDICE "Flag to turn on LandIce Source code" OFF)
OPTION(ENABLE_CISM_INTERFACE "Flag to turn on LandIce Interface to the CISM code" OFF)
OPTION(ENABLE_MPAS_INTERFACE "Flag to turn on LandIce Source code" OFF)
IF (ENABLE_LANDICE)
  MESSAGE("-- LandIce                         is Enabled")
  SET(ALBANY_LANDICE TRUE)

  # set optional dependency of mesh on parameters, defaults to Disabled
  OPTION(ENABLE_USE_CISM_FLOW_PARAMETERS "Flag to use the parameters of Glen's law used by Cism" ON)
  IF (ENABLE_USE_CISM_FLOW_PARAMETERS)
    MESSAGE("  -- USE_CISM_FLOW_PARAMETERS      is Enabled.")
    SET(USE_CISM_FLOW_PARAMETERS TRUE)
  ELSE()
    SET(USE_CISM_FLOW_PARAMETERS FALSE)
    MESSAGE("  -- USE_CISM_FLOW_PARAMETERS      is NOT Enabled.")
  ENDIF()

  # Set optional build of MPAS Interface for LandIce (Land Ice SciDAC project),
  # defaults to Disabled
  IF (ENABLE_MPAS_INTERFACE)
    MESSAGE("-- MPAS_INTERFACE                  is Enabled")
    SET(MPAS_HAS_LANDICE TRUE)
  ELSE()
    MESSAGE("  -- MPAS_INTERFACE                is NOT Enabled.")
    SET(MPAS_HAS_LANDICE FALSE)
  ENDIF()

  # Set optional build of CISM Interface for LandIce (Land Ice SciDAC project),
  # defaults to Disabled
  OPTION(ENABLE_CISM_REDUCED_COMM "Flag to turn on reduced communicator for CISM-Albany runs" ON)
  OPTION(ENABLE_CISM_CHECK_COMPARISONS "Flag to turn on comparisons checking for CISM-Albany" OFF)

  IF (ENABLE_CISM_INTERFACE)
    MESSAGE("  -- CISM_INTERFACE                is Enabled")
    SET(CISM_HAS_LANDICE TRUE)
    IF (ENABLE_CISM_CHECK_COMPARISONS)
      SET(CISM_CHECK_COMPARISONS TRUE)
      MESSAGE("  -- CISM_CHECK_COMPARISONS      is Enabled. Error will be thrown if comparisons fail for CISM-Albany run.")
    ELSE()
      MESSAGE("  -- CISM_CHECK_COMPARISONS      is Disabled.  Error will NOT be thrown if comparisons fail for CISM-Albany run.  Please be cautious when running ctest from CISM!")
    ENDIF()
    IF (ENABLE_CISM_REDUCED_COMM)
      SET(REDUCED_COMM TRUE)
      MESSAGE("  -- CISM_REDUCED_COMM           is Enabled.  CISM-Albany will use reduced communicator.")
    ENDIF()
    SET(CISM_HAS_LANDICE TRUE)
    IF (NOT CISM_INCLUDE_DIR)
      MESSAGE("\nError:  CISM_INCLUDE_DIR not defined\n")
    ELSE()
      MESSAGE("  -- CISM_INCLUDE_DIR ${CISM_INCLUDE_DIR}")
    ENDIF()
    IF (NOT CISM_EXE_DIR)
      MESSAGE("\n CISM_EXE_DIR not defined.  CismAlbany test will not run.\n")
    ELSE()
      MESSAGE("  -- CISM_EXE_DIR ${CISM_EXE_DIR}")
    ENDIF()
  ELSE()
    MESSAGE("  -- CISM_INTERFACE                is NOT Enabled.")
    SET(CISM_HAS_LANDICE FALSE)
  ENDIF()


ELSE()
  MESSAGE("-- LandIce                         is NOT Enabled.")
  SET(ALBANY_LANDICE FALSE)
  IF(ENABLE_MPAS_INTERFACE)
    MESSAGE(FATAL_ERROR "\nError:  ENABLE_MPAS_INTERFACE requires ENABLE_LANDICE
  to be ON\n")
  ELSEIF(ENABLE_CISM_INTERFACE)
    MESSAGE(FATAL_ERROR "\nError:  ENABLE_CISM_INTERFACE requires ENABLE_LANDICE
  to be ON\n")
  ENDIF()
ENDIF()

OPTION(ENABLE_TSUNAMI "Flag to turn on TSUNAMI Source code" OFF)
IF (ENABLE_TSUNAMI)
  MESSAGE("-- TSUNAMI                         is Enabled")
  SET(ALBANY_TSUNAMI TRUE)
ELSE()
  MESSAGE("-- TSUNAMI                         is NOT Enabled.")
  SET(ALBANY_TSUNAMI FALSE)
ENDIF()

# set optional dependency of mesh on parameters, defaults to Disabled
OPTION(ENABLE_MESH_DEPENDS_ON_PARAMETERS "Flag to turn on dependency of mesh on parameters, e.g for shape optimization" OFF)
IF (ENABLE_MESH_DEPENDS_ON_PARAMETERS)
  MESSAGE("-- MESH_DEPENDS_ON_PARAMETERS      is Enabled")
  SET(ALBANY_MESH_DEPENDS_ON_PARAMETERS TRUE)
ELSE()
  MESSAGE("-- MESH_DEPENDS_ON_PARAMETERS      is NOT Enabled.")
  SET(ALBANY_MESH_DEPENDS_ON_PARAMETERS FALSE)
ENDIF()

# set optional dependency of mesh on solution, defaults to Disabled
OPTION(ENABLE_MESH_DEPENDS_ON_SOLUTION "Flag to turn on dependency of mesh on solution" OFF)
IF (ENABLE_MESH_DEPENDS_ON_SOLUTION)
  MESSAGE("-- MESH_DEPENDS_ON_SOLUTION        is Enabled")
  SET(ALBANY_MESH_DEPENDS_ON_SOLUTION TRUE)
ELSE()
  MESSAGE("-- MESH_DEPENDS_ON_SOLUTION        is NOT Enabled.")
  SET(ALBANY_MESH_DEPENDS_ON_SOLUTION FALSE)
ENDIF()

# set optional dependency of dist parameters on solution, defaults to Disabled
OPTION(ENABLE_PARAMETERS_DEPEND_ON_SOLUTION "Flag to turn on dependency of mesh on solution" OFF)
IF (ENABLE_PARAMETERS_DEPEND_ON_SOLUTION)
  MESSAGE("-- PARAMETERS_DEPEND_ON_SOLUTION   is Enabled")
  SET(ALBANY_PARAMETERS_DEPEND_ON_SOLUTION TRUE)
ELSEIF (ENABLE_MESH_DEPENDS_ON_SOLUTION)
  MESSAGE("-- PARAMETERS_DEPEND_ON_SOLUTION   is Enabled because MESH_DEPENDS_ON_SOLUTION is enabled")
  SET(ALBANY_PARAMETERS_DEPEND_ON_SOLUTION TRUE)
ELSE()
  MESSAGE("-- PARAMETERS_DEPEND_ON_SOLUTION   is NOT Enabled.")
  SET(ALBANY_PARAMETERS_DEPEND_ON_SOLUTION FALSE)
ENDIF()

# Set optional build of Aeras (Atmosphere Dynamics LDRD), defaults to Disabled
OPTION(ENABLE_AERAS "Flag to turn on Aeras Source code" OFF)
OPTION(ENABLE_AERAS_IMPLICIT_HS "Flag to turn on implicit time-int scheme for Aeras hydrostatic Source code" OFF)
IF (ENABLE_AERAS)
  MESSAGE("-- Aeras                           is Enabled")
  SET(ALBANY_AERAS TRUE)
  IF (ENABLE_AERAS_IMPLICIT_HS)
    MESSAGE("  -- Aeras implicit hydrostatic    is Enabled")
    SET(AERAS_IMPLICIT_HS TRUE)
  ELSE()
    MESSAGE("  -- Aeras implicit hydrostatic    is NOT Enabled")
    SET(AERAS_IMPLICIT_HS FALSE)
  ENDIF()
ELSE()
  MESSAGE("-- Aeras                           is NOT Enabled.")
  SET(ALBANY_AERAS FALSE)
ENDIF()

# optional dependency on LAME, defaults to FALSE
OPTION(ENABLE_LAME "Flag to turn on LAME material library source code" OFF)
IF(ENABLE_LAME)
  MESSAGE("-- LAME                            is enabled")
  SET(ALBANY_LAME TRUE)
  # The Sierra math toolkit is needed for LAME
  IF(NOT MATH_TOOLKIT_INCLUDE_DIR)
    MESSAGE("\nError:  MATH_TOOLKIT_INCLUDE_DIR not defined\n")
  ELSE()
    MESSAGE("  -- MATH_TOOLKIT_INCLUDE_DIR ${MATH_TOOLKIT_INCLUDE_DIR}")
  ENDIF()
  IF(NOT MATH_TOOLKIT_LIBRARY_DIR)
    MESSAGE("\nError:  MATH_TOOLKIT_LIBRARY_DIR not defined\n")
  ELSE()
    MESSAGE("  -- MATH_TOOLKIT_LIBRARY_DIR ${MATH_TOOLKIT_LIBRARY_DIR}")
  ENDIF()
  # LAME itself
  IF(NOT LAME_INCLUDE_DIR)
    MESSAGE("\nError:  LAME_INCLUDE_DIR not defined\n")
  ELSE()
    MESSAGE("  -- LAME_INCLUDE_DIR ${LAME_INCLUDE_DIR}")
  ENDIF()
  IF(NOT LAME_LIBRARY_DIR)
    MESSAGE("\nError:  LAME_LIBRARY_DIR not defined\n")
  ELSE()
    MESSAGE("  -- LAME_LIBRARY_DIR ${LAME_LIBRARY_DIR}")
  ENDIF()
ELSE()
  MESSAGE("-- LAME                            is NOT Enabled.")
  SET(ALBANY_LAME FALSE)
ENDIF()

# optional dependency on LAMENT, defaults to FALSE
OPTION(ENABLE_LAMENT "Flag to turn on LAMENT lmaterial library source code" OFF)
IF(ENABLE_LAMENT)
  MESSAGE("-- LAMENT                          is enabled")
  SET(ALBANY_LAMENT TRUE)
  IF(NOT LAMENT_INCLUDE_DIR)
    MESSAGE("\nError:  LAMENT_INCLUDE_DIR not defined\n")
  ELSE()
    MESSAGE("  -- LAMENT_INCLUDE_DIR ${LAMENT_INCLUDE_DIR}")
  ENDIF()
  IF(NOT LAMENT_LIBRARY_DIR)
    MESSAGE("\nError:  LAMENT_LIBRARY_DIR not defined\n")
  ELSE()
    MESSAGE("  -- LAMENT_LIBRARY_DIR ${LAMENT_LIBRARY_DIR}")
  ENDIF()
ELSE()
  MESSAGE("-- LAMENT                          is NOT Enabled.")
  SET(ALBANY_LAMENT FALSE)
ENDIF()

# Add option for ASCR work, defaults to disabled
OPTION(ENABLE_ASCR "Flag to turn on ASCR-funded coupled UQ research source code" OFF)
IF (ENABLE_ASCR)
  MESSAGE("-- ASCR                            is Enabled")
  SET(ALBANY_ASCR TRUE)
ELSE()
  MESSAGE("-- ASCR                            is NOT Enabled.")
  SET(ALBANY_ASCR FALSE)
ENDIF()

# Set optional dependency on AMP, defaults to Disabled
OPTION(ENABLE_AMP "Flag to turn on AMP (additive manufacturing processes) source code" OFF)
IF (ENABLE_AMP)
  SET(ALBANY_APF TRUE)
  MESSAGE("-- AMP                             is Enabled")
  SET(ALBANY_AMP TRUE)
ELSE()
  MESSAGE("-- AMP                             is NOT Enabled.")
  SET(ALBANY_AMP FALSE)
ENDIF()

# Set optional dependency on SCOREC PUMI,  optional adaptivity
OPTION(ENABLE_SCOREC "Flag to turn on SCOREC dependency" OFF)
SET(ALBANY_SCOREC_DIR "" CACHE PATH "Path to standalone SCOREC installation")
IF(ENABLE_SCOREC)
  # Check whether SCOREC was built as part of Trilinos
  LIST(FIND Trilinos_PACKAGE_LIST SCOREC SCOREC_List_ID)
  IF (SCOREC_List_ID GREATER -1)
    # It was built in Trilinos
    LIST(FIND Trilinos_TPL_LIST SimModel SimModel_List_ID)
    # Check whether Simmetrix is also used
    IF (SimModel_List_ID GREATER -1)
      MESSAGE(STATUS "Simmetrix SimModel is a Trilinos TPL")
      SET(ALBANY_SCOREC_SIMMODEL ON)
    ENDIF()
    SET(ALBANY_SCOREC_STANDALONE FALSE)
  ELSE()
    # Not built in Trilinos, try to find standalone
    # build in SCOREC_PREFIX
    FIND_PACKAGE(SCOREC 2.0.0 PATHS "${ALBANY_SCOREC_DIR}")
    IF (NOT SCOREC_FOUND)
      MESSAGE(FATAL_ERROR "\nError: can't find either SCOREC package\n")
    ENDIF()
    MESSAGE(STATUS "SCOREC Found at ${SCOREC_CONFIG}")
    SET(ALBANY_SCOREC_STANDALONE TRUE)
  ENDIF()
  SET(ALBANY_APF TRUE)
  MESSAGE("-- SCOREC                          is enabled")
  SET(ALBANY_SCOREC TRUE)
ELSE()
  MESSAGE("-- SCOREC                          is NOT Enabled.")
  SET(ALBANY_SCOREC FALSE)
ENDIF()

# RPI APF mesh interface
IF (ALBANY_APF)
  MESSAGE("-- APF                             is Enabled")
ELSE()
  MESSAGE("-- APF                             is NOT Enabled.")
ENDIF()

OPTION(ENABLE_OMEGA_H "Enable Omega_h dependency for adaptation" OFF)
SET(ALBANY_OMEGA_H_DIR "" CACHE PATH "Path to Omega_h installation")
IF(ENABLE_OMEGA_H)
  FIND_PACKAGE(Omega_h 9.0.0 REQUIRED PATHS "${ALBANY_OMEGA_H_DIR}")
  MESSAGE(STATUS "Omega_h found at ${Omega_h_CONFIG}")
  SET(ALBANY_OMEGA_H TRUE)
ENDIF()

# ParaView's Catalyst CoProcessing package
OPTION(ENABLE_CATALYST "Flag to turn on support for the Catalyst CoProcessing library" OFF)
IF (ENABLE_CATALYST)
  MESSAGE("-- Catalyst                        is Enabled")
  SET(ALBANY_CATALYST TRUE)
ELSE()
  MESSAGE("-- Catalyst                        is NOT Enabled.")
  SET(ALBANY_CATALYST FALSE)
ENDIF()

# Add option for CONTACT work, defaults to disabled
OPTION(ENABLE_CONTACT "Flag to turn on CONTACT code in LCM" OFF)
IF (ENABLE_CONTACT)
  MESSAGE("-- CONTACT                         is Enabled")
  SET(ALBANY_CONTACT TRUE)
ELSE()
  MESSAGE("-- CONTACT                         is NOT Enabled.")
  SET(ALBANY_CONTACT FALSE)
ENDIF()

## Other Albany Template Options here:
MESSAGE("\nAlbany Templated Data Type Options:")

# Throw error if using ENABLE_FAST_FELIX
IF(ENABLE_FAST_FELIX)
  MESSAGE(FATAL_ERROR
  "\nError: ENABLE_FAST_FELIX has been removed! Use ENABLE_FAD_TYPE instead.
  Example for a tetrahedral mesh in LandIce:
    -D ENABLE_FAD_TYPE:STRING=SFad
    -D ALBANY_SFAD_SIZE=8
  For a hexahedral mesh, use ALBANY_SFAD_SIZE=16")
ENDIF()

# Set FAD data type
SET(ENABLE_FAD_TYPE "DFad" CACHE STRING "Sacado forward mode automatic differentiation data type")
IF(ENABLE_FAD_TYPE STREQUAL "SFad")
  SET(ALBANY_FAD_TYPE_SFAD TRUE)
  SET(ALBANY_SFAD_SIZE 32 CACHE INT "Number of derivative components chosen at compile-time for AD")
  MESSAGE("-- FAD_TYPE  is SFad, ALBANY_SFAD_SIZE=${ALBANY_SFAD_SIZE}")
  MESSAGE("---> WARNING: problems with elemental DOFs > ${ALBANY_SFAD_SIZE} will fail")
ELSEIF(ENABLE_FAD_TYPE STREQUAL "SLFad")
  SET(ALBANY_FAD_TYPE_SLFAD TRUE)
  SET(ALBANY_SLFAD_SIZE 32 CACHE INT "Maximum number of derivative components chosen at compile-time for AD")
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
  SET(ALBANY_TAN_SFAD_SIZE 32 CACHE INT "Number of derivative components chosen at compile-time for Tangent AD")
  MESSAGE("-- TAN_FAD_TYPE is SFad, ALBANY_TAN_SFAD_SIZE=${ALBANY_TAN_SFAD_SIZE}")
  MESSAGE("---> WARNING: problems with Tangent elemental DOFs > ${ALBANY_TAN_SFAD_SIZE} will fail")
ELSEIF(ENABLE_TAN_FAD_TYPE STREQUAL "SLFad")
  SET(ALBANY_TAN_FAD_TYPE_SLFAD TRUE)
  SET(ALBANY_TAN_SLFAD_SIZE 32 CACHE INT "Maximum number of derivative components chosen at compile-time for Tangent AD")
  MESSAGE("-- TAN_FAD_TYPE is SLFad, ALBANY_TAN_SLFAD_SIZE=${ALBANY_TAN_SLFAD_SIZE}")
  MESSAGE("---> WARNING: problems with Tangent elemental DOFs > ${ALBANY_TAN_SLFAD_SIZE} will fail")
ELSEIF(ENABLE_TAN_FAD_TYPE STREQUAL "DFad")
  MESSAGE("-- TAN_FAD_TYPE is DFad (default)")
ELSE()
  MESSAGE(FATAL_ERROR
  "\nError: ENABLE_TAN_FAD_TYPE = ${ENABLE_TAN_FAD_TYPE} is not recognized!
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

# optionally disable the use of the Trilinos stokhos package
OPTION(ENABLE_STOKHOS "Flag to enable / disable the use of Stokhos in Albany" OFF)
IF (ENABLE_STOKHOS)
  MESSAGE("-- Stokhos   is Enabled")
  set(ALBANY_STOKHOS TRUE)
  LIST(FIND Trilinos_PACKAGE_LIST Stokhos STOKHOS_List_ID)
  IF (NOT STOKHOS_List_ID GREATER -1)
    MESSAGE(FATAL_ERROR "\nError: STOKHOS option requires Stokhos\n")
  ENDIF()
ELSE()
  set(ALBANY_STOKHOS FALSE)
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
MESSAGE("\nMiscellaneous Albany Options:")

# It is not currently possible to determine the status of "Kokkos_ENABLE_Cuda" from Albany. So we will deduce it
# from ${Trilinos_INCLUDE_DIRS}/KokkosCore_config.h
IF(NOT DEFINED Kokkos_ENABLE_Cuda)
  SET(Kokkos_ENABLE_Cuda OFF)
  IF(EXISTS "${Trilinos_INCLUDE_DIRS}/KokkosCore_config.h")
    FILE(READ ${Trilinos_INCLUDE_DIRS}/KokkosCore_config.h CURRENT_CONFIG)
    STRING(REGEX MATCH "\#define KOKKOS_ENABLE_CUDA" KOKKOS_CUDA_IS_SET ${CURRENT_CONFIG})
    IF("#define KOKKOS_ENABLE_CUDA" STREQUAL "${KOKKOS_CUDA_IS_SET}")
      MESSAGE("-- Kokkos is configured to use CUDA, Albany will also.")
      SET(Kokkos_ENABLE_Cuda ON)
      IF (ENABLE_LCM) 
        MESSAGE(FATAL_ERROR "\nError:  LCM does not work with a CUDA KokkosNode!  Recompile with LCM off.")
      ENDIF()
      SET(NUM_GPUS_PER_NODE 4 CACHE INT "set Number of GPUs per Node")
      MESSAGE("-- NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}")
      SET(KOKKOS_NDEVICES "--kokkos-ndevices=${NUM_GPUS_PER_NODE}")
      IF (ENABLE_CUDA_ERROR_CHECK)
        SET(ALBANY_CUDA_ERROR_CHECK ON)
        MESSAGE("-- CUDA Error Check        is Enabled")
      ELSE()
        MESSAGE("-- CUDA Error Check        is NOT Enabled.")
      ENDIF()
      IF (ENABLE_ALBANY_EPETRA)
        MESSAGE(FATAL_ERROR "\nError:  Albany with Epetra is not supported with CUDA KokkosNode!  Please re-configure w/o Epetra (ENABLE_ALBANY_EPETRA=OFF).")
      ENDIF()
      IF (ENABLE_CUDA_NVTX)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lnvToolsExt")
        SET(ALBANY_CUDA_NVTX ON)
        MESSAGE("-- CUDA NVTX               is Enabled")
      ELSE()
        MESSAGE("-- CUDA NVTX               is NOT Enabled.")
      ENDIF()
    ELSE("#define KOKKOS_ENABLE_CUDA" STREQUAL "${KOKKOS_CUDA_IS_SET}")
      MESSAGE("-- Kokkos is not configured to use CUDA, Albany will NOT be compiled for CUDA.")
    ENDIF("#define KOKKOS_ENABLE_CUDA" STREQUAL "${KOKKOS_CUDA_IS_SET}")
  ENDIF(EXISTS "${Trilinos_INCLUDE_DIRS}/KokkosCore_config.h")
ENDIF(NOT DEFINED Kokkos_ENABLE_Cuda)

# It is not currently possible to determine the status of "Kokkos_ENABLE_Cuda" from Albany. So we will deduce it
# from ${Trilinos_INCLUDE_DIRS}/KokkosCore_config.h
IF(NOT DEFINED Kokkos_ENABLE_OpenMP)
  SET(Kokkos_ENABLE_OpenMP OFF)
  IF(EXISTS "${Trilinos_INCLUDE_DIRS}/KokkosCore_config.h")
    FILE(READ ${Trilinos_INCLUDE_DIRS}/KokkosCore_config.h CURRENT_CONFIG)
    STRING(REGEX MATCH "\#define KOKKOS_ENABLE_OPENMP" KOKKOS_OPENMP_IS_SET ${CURRENT_CONFIG})
    IF("#define KOKKOS_ENABLE_OPENMP" STREQUAL "${KOKKOS_OPENMP_IS_SET}")
      MESSAGE("-- Kokkos is configured to use OpenMP, Albany will also.")
      SET(Kokkos_ENABLE_OpenMP ON)
      IF (ENABLE_LARGE_TESTS AND ENABLE_LCM) 
        MESSAGE("   -> LCM WARNING: turning off OrientationOnMesh tests since KokkosNode = OpenMP (see issue #339)!")
      ENDIF(ENABLE_LARGE_TESTS AND ENABLE_LCM)
      IF (ENABLE_LCM) 
        MESSAGE("   -> LCM WARNING: turning off ACE tests since KokkosNode = OpenMP (see issue #501)!")
      ENDIF(ENABLE_LCM)
    ELSE("#define KOKKOS_ENABLE_OPENMP" STREQUAL "${KOKKOS_OpenMP_IS_SET}")
      MESSAGE("-- Kokkos is not configured to use OpenMP, Albany will NOT be compiled for OpenMP.")
    ENDIF("#define KOKKOS_ENABLE_OPENMP" STREQUAL "${KOKKOS_OPENMP_IS_SET}")
  ENDIF(EXISTS "${Trilinos_INCLUDE_DIRS}/KokkosCore_config.h")
ENDIF(NOT DEFINED Kokkos_ENABLE_OpenMP)

# set optional dependency on the BGL, defaults to Enabled
# This option is added due to issued with compiling BGL with the intel compilers
# see Trilinos bugzilla bug #6343
#
OPTION(ENABLE_BGL "Flag to turn on the Phalanx graphviz capabilities (dependent on BGL)" ON)
IF (ENABLE_BGL)
  MESSAGE("-- BGL                       is Enabled")
  SET(ALBANY_BGL TRUE)
ELSE()
  MESSAGE("-- BGL                       is NOT Enabled. Graph visualization in Phalanx will not be available!")
  SET(ALBANY_BGL FALSE)
ENDIF()

# Add option to check for hardware floating point exceptions using SSE, defaults to disabled
OPTION(ENABLE_CHECK_FPE "Flag to turn on floating point exception (NAN) checking" OFF)

# Add option to check for hardware floating point exceptions using SSE, defaults to disabled
OPTION(ENABLE_CHECK_FPE "Flag to turn on floating point exception (NAN) checking" OFF)
IF (ENABLE_CHECK_FPE)
  MESSAGE("-- FPE checking              is Enabled")
  SET(ALBANY_CHECK_FPE TRUE)
ELSE()
  MESSAGE("-- FPE checking              is NOT Enabled.")
  SET(ALBANY_CHECK_FPE FALSE)
ENDIF()

OPTION(ENABLE_STRONG_FPE_CHECK "Flag to turn on strong checking for floating point exceptions" OFF)
IF (ENABLE_STRONG_FPE_CHECK)
  MESSAGE("-- Strong FPE checking       is Enabled")
  SET(ALBANY_STRONG_FPE_CHECK TRUE)
ELSE()
  MESSAGE("-- Strong FPE checking       is NOT Enabled.")
  SET(ALBANY_STRONG_FPE_CHECK FALSE)
ENDIF()

# Add option to set FTZ and DAZ
OPTION(ENABLE_FLUSH_DENORMALS "Flag to turn on FTZ and DAZ" OFF)
IF (ENABLE_FLUSH_DENORMALS)
  MESSAGE("-- FTZ/DAZ flags             is Enabled")
  SET(ALBANY_FLUSH_DENORMALS TRUE)
ELSE()
  MESSAGE("-- FTZ/DAZ flags             is NOT Enabled.")
  SET(ALBANY_FLUSH_DENORMALS FALSE)
ENDIF()

#amb Default to on until the merge.
OPTION(ENABLE_KOKKOS_UNDER_DEVELOPMENT "Flag to turn on Kokkos experimental code in Albany" ON)
IF (ENABLE_KOKKOS_UNDER_DEVELOPMENT)
  MESSAGE("-- Kokkos development        is Enabled")
  SET(ALBANY_KOKKOS_UNDER_DEVELOPMENT TRUE)
ELSE()
  MESSAGE("-- Kokkos development        is NOT Enabled.")
  SET(ALBANY_KOKKOS_UNDER_DEVELOPMENT FALSE)
ENDIF()

OPTION(ENABLE_ALBANY_VERBOSE "Flag to turn on verbose output" OFF)
IF (NOT ENABLE_ALBANY_VERBOSE)
  MESSAGE("-- Verbose printing          is NOT Enabled")
  SET(ALBANY_VERBOSE FALSE)
ELSE()
  MESSAGE("-- Verbose printing          is Enabled")
  SET(ALBANY_VERBOSE TRUE)
ENDIF()

# Option to enable "speculative" features that are not ready for general visibility
OPTION(ENABLE_LCM_SPECULATIVE "Flag to turn on speculative LCM features/code" OFF)
IF (NOT ENABLE_LCM_SPECULATIVE)
  MESSAGE("-- Speculative LCM features  are NOT Enabled")
  SET(LCM_SPECULATIVE FALSE)
ELSE()
  MESSAGE("-- Speculative LCM features  are Enabled")
  SET(LCM_SPECULATIVE TRUE)
ENDIF()

# Enhanced debugging option
OPTION(ENABLE_DEBUGGING "Flag to turn on enhanced debugging info" OFF)
IF (ENABLE_DEBUGGING)
  MESSAGE("-- Enhanced debugging        is Enabled")
  SET(ALBANY_DEBUG TRUE)
ELSE()
  MESSAGE("-- Enhanced debugging        is NOT Enabled.")
  SET(ALBANY_DEBUG FALSE)
ENDIF()

#Add option to build libraries but not link any executables (for LandIce uner MPAS)
OPTION(ALBANY_LIBRARIES_ONLY "Flag to only build libraries and not link executables" OFF)
IF (ALBANY_LIBRARIES_ONLY)
  MESSAGE("-- ALBANY_LIBRARIES_ONLY is ON")
  IF (LCM_TEST_EXES)
    SET (LCM_TEST_EXES FALSE)
    MESSAGE("--->  Disabling LCM_TEST_EXES")
  ENDIF()
ENDIF()

# Add option to include Dakota restart tests in the test suite
IF (NOT APPLE)
  CMAKE_DEPENDENT_OPTION(ENABLE_DAKOTA_RESTART_EXAMPLES "Flag to include Dakota restart tests" ON
                         "ALBANY_DAKOTA" OFF)
ELSE (NOT APPLE)
  # Portable Dakota restart files is broken on MacOS, deactivate test by default
  CMAKE_DEPENDENT_OPTION(ENABLE_DAKOTA_RESTART_EXAMPLES "Flag to include Dakota restart tests" OFF
                         "ALBANY_DAKOTA" OFF)
ENDIF (NOT APPLE)
IF (ENABLE_DAKOTA_RESTART_EXAMPLES)
  MESSAGE("-- Dakota restart testing    is Enabled")
  SET (ALBANY_DAKOTA_RESTART_EXAMPLES TRUE)
ELSE (ENABLE_DAKOTA_RESTART_EXAMPLES)
  IF (ALBANY_DAKOTA)
    MESSAGE("-- Dakota restart testing    is NOT Enabled")
  ENDIF (ALBANY_DAKOTA)
  SET (ALBANY_DAKOTA_RESTART_EXAMPLES FALSE)
ENDIF (ENABLE_DAKOTA_RESTART_EXAMPLES)

# Set which memory analysis system routines are available.
option(ENABLE_MALLINFO "Use mallinfo() in Albany::printMemoryAnalysis()" off)
option(ENABLE_GETRUSAGE "Use getrusage() in Albany::printMemoryAnalysis()" off)
option(ENABLE_KERNELGETMEMORYSIZE "Use Kernel_GetMemorySize() in Albany::printMemoryAnalysis()" off)
if (ENABLE_MALLINFO)
  SET(ALBANY_HAVE_MALLINFO TRUE)
  message("-- Memory: mallinfo()        is Enabled.")
endif()
if (ENABLE_GETRUSAGE)
  SET(ALBANY_HAVE_GETRUSAGE TRUE)
  message("-- Memory: getrusage()       is Enabled.")
endif()
if (ENABLE_KERNELGETMEMORYSIZE)
  SET(ALBANY_HAVE_KERNELGETMEMORYSIZE ON)
  message("-- Memory: Kernel_GetMemorySize() is Enabled.")
endif()

# Mesh database tools.
OPTION(ENABLE_MESHDB_TOOLS "Flag to turn on mesh database tools" OFF)
IF (ENABLE_MESHDB_TOOLS)
  MESSAGE("-- Mesh database tools       are Enabled.")
  SET(ALBANY_MESHDB_TOOLS TRUE)
ENDIF()

# Repeat this warning.

IF (TriKota_List_ID GREATER -1 AND NOT ENABLE_ALBANY_EPETRA)
  MESSAGE(FATAL_ERROR "\nError:  TriKota option does not work with
ENABLE_ALBANY_EPETRA turned off!  Turn on ENABLE_ALBANY_EPETRA if you
want to useTriKota.")
ENDIF (TriKota_List_ID GREATER -1 AND NOT ENABLE_ALBANY_EPETRA)

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

## End of CMake options to Albany

add_subdirectory(src)

MESSAGE ("\n ***********************")
MESSAGE (" *** Testing Options ***")
MESSAGE (" ***********************\n")

# Add option to enable performance tests. Results are machine-specific
OPTION(ENABLE_PERFORMANCE_TESTS "Flag to turn on Performance Tests" OFF)
IF (ENABLE_PERFORMANCE_TESTS)
  MESSAGE("-- Performance Tests         are Enabled, adding tests/*/PerformanceTests")
  SET(ALBANY_PERFORMANCE_TESTS TRUE)
ELSE()
  MESSAGE("-- Performance Tests         are NOT Enabled.")
  SET(ALBANY_PERFORMANCE_TESTS FALSE)
ENDIF()


# Add option to enable performance tests. Results are machine-specific
OPTION(ENABLE_LARGE_TESTS "Run tests from the Albany-large-tests repository" ON)
IF (ENABLE_LARGE_TESTS)
  MESSAGE("-- Large Tests               are Enabled, adding tests/large/")
  SET(ALBANY_LARGE_TESTS TRUE)
ELSE()
  MESSAGE("-- Large Tests               are NOT Enabled.")
  SET(ALBANY_LARGE_TESTS FALSE)
ENDIF()


# Set option to disable compiling some code for nightly tests
# Some code sections take so long to compile with NVCC that nightly tests are
# timing out. This option allows disabling those sections.
OPTION(NIGHTLY_TESTING "Flag to turn off selected code during nightly testing" OFF)
IF (NIGHTLY_TESTING)
  MESSAGE("-- NIGHTLY_TESTING           is Enabled")
  SET(ALBANY_NIGHTLY_TESTING TRUE)
ELSE()
  MESSAGE("-- NIGHTLY_TESTING           is NOT Enabled.")
  SET(ALBANY_NIGHTLY_TESTING FALSE)
ENDIF()

#  Timeout real long running tests
IF(ALBANY_CTEST_TIMEOUT)
    MESSAGE("\n-- Setting maximum test run time to be       ${ALBANY_CTEST_TIMEOUT} seconds")
ELSE()
    MESSAGE("\n-- No limits are set on maximum test run time")
ENDIF()

# Make a clean build target for convenience

ADD_CUSTOM_TARGET( cmake_clean
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/Makefile
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/cmake_install.cmake
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake
)

OPTION(ALBANY_PARALLEL_ONLY "Only run the parallel tests" OFF)

IF(ALBANY_MPI)
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
ENDIF(ALBANY_MPI)

#Add option to run any pretest executables in a specific environment (mpiexec.hydra, srun, or other)
IF (DEFINED ALBANY_PRETEST_EXEC)
  MESSAGE("-- ALBANY_PRETEST_EXEC is ${ALBANY_PRETEST_EXEC}")
ELSE()
  # Not set in configure - set it to be empty
  SET(ALBANY_PRETEST_EXEC "" CACHE STRING "Command to use to execute pretests")
ENDIF()

#Add option to specify path to desired SEACAS utils
IF (DEFINED ALBANY_SEACAS_PATH)
  MESSAGE("-- ALBANY_SEACAS_PATH is ${ALBANY_SEACAS_PATH}")
ENDIF()

# Add Albany test -- in the "tests" directory
IF (ALBANY_LIBRARIES_ONLY)
  MESSAGE("\nALBANY_LIBRARIES_ONLY has been set: turning off all tests.")
ELSE()
  add_subdirectory(tests)
ENDIF()

MESSAGE("\nAnd CMake says...:")
