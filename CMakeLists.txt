##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# CMAKE File for Albany building against an installed Trilinos

cmake_minimum_required(VERSION 2.7)
include(CMakeDependentOption)

# Error check up front
#IF (NOT DEFINED ALBANY_TRILINOS_DIR)
#  MESSAGE(FATAL_ERROR "\nAlbany Error: cmake must define ALBANY_TRILINOS_DIR:
#  (-D ALBANY_TRILINOS_DIR:FILEPATH=<trilinos_install_prefix>)!")
#ENDIF()

# Advertise that we support shared libs
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

# Get Trilinos as one entity
IF (DEFINED ALBANY_TRILINOS_DIR)
  SET(CMAKE_PREFIX_PATH ${ALBANY_TRILINOS_DIR} ${CMAKE_PREFIX_PATH})
ENDIF()
FIND_PACKAGE(Trilinos REQUIRED)

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

# Optional Installation helpers
# Note that some of this functionality depends on CMAKE > 2.8.8
SET (INSTALL_ALBANY FALSE)
IF (ENABLE_INSTALL)
  IF (${CMAKE_VERSION} VERSION_GREATER 2.8.8 OR ${CMAKE_VERSION} VERSION_EQUAL 2.8.8)
    SET (INSTALL_ALBANY TRUE)
    MESSAGE("")
    MESSAGE("Installation files for Albany will be created")
    MESSAGE("-- install location = ${CMAKE_INSTALL_PREFIX}")
    MESSAGE("")
  ELSE()
    MESSAGE("")
    MESSAGE("Installation has been requested, but cannot continue")
    MESSAGE("-- Please upgrade CMAKE to version >= 2.8.8")
    MESSAGE("")
  ENDIF()
ENDIF()

IF (INSTALL_ALBANY)
  set(BINARY_INSTALL_DIR bin)
  set(INCLUDE_INSTALL_DIR include)
  set(LIB_INSTALL_DIR lib)
  include(CMakePackageConfigHelpers)
ENDIF ()

MESSAGE("Setting and checking of compilers:")
SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER} )
SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER} )
SET(CMAKE_Fortran_COMPILER ${Trilinos_Fortran_COMPILER} )

IF (CMAKE_Fortran_COMPILER) # Enable Fortran if it is enabled in Trilinos.
  ENABLE_LANGUAGE(Fortran)
ENDIF()

# Build Albany as shared libraries if Trilinos was compiled that way

IF(Trilinos_BUILD_SHARED_LIBS)
  SET(Albany_LIBRARY_TYPE SHARED)
  SET(BUILD_SHARED_LIBS ON)
  MESSAGE("-- Building Albany with shared libraries")
ELSE(Trilinos_BUILD_SHARED_LIBS)
  SET(Albany_LIBRARY_TYPE STATIC)
ENDIF(Trilinos_BUILD_SHARED_LIBS)

# End of setup and error checking
#  NOTE: PROJECT command checks for compilers, so this statement
#        is moved AFTER setting CMAKE_CXX_COMPILER from Trilinos

PROJECT(Albany)

IF (NOT CMAKE_CXX_FLAGS)
  SET(CMAKE_CXX_FLAGS ${Trilinos_CXX_COMPILER_FLAGS} )
ENDIF()
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ALBANY_CXX_FLAGS}")

IF(${CMAKE_CXX_COMPILER_ID} MATCHES "Intel")
  SET(USING_INTEL TRUE)
# On Intel, turn of interprocedural optimization at linking to keep linker from hanging
# This could possibly be removed with further study (GAH)
#  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-ip")
   SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd2536,2196,858,47")
ENDIF()

## Check for some required Trilinos package dependencies (Piro, Phalanx) ##
MESSAGE("\nChecking Trilinos build for required and optional packages:")

## Could check for them all, just starting with two non-standard ones ##
LIST(FIND Trilinos_PACKAGE_LIST Piro Piro_List_ID)
IF (Piro_List_ID GREATER -1)
  MESSAGE("-- Found Piro,    a required Trilinos package for Albany.")
ELSE()
  MESSAGE("-- FATAL ERROR: Did not find Piro package in Trilinos Installation.")
  MESSAGE(FATAL_ERROR "-- Configure Trilinos with guidance from trilinos-cmake script in doc directory.")
ENDIF()
LIST(FIND Trilinos_PACKAGE_LIST Phalanx Phalanx_List_ID)
IF (Phalanx_List_ID GREATER -1)
  MESSAGE("-- Found Phalanx, a required Trilinos package for Albany.")
ELSE()
  MESSAGE("-- FATAL ERROR: Did not find Phalanx package in Trilinos Installation.")
  MESSAGE(FATAL_ERROR "-- Configure Trilinos with guidance from trilinos-cmake script in doc directory.")
ENDIF()

### Check for some OPTIONAL trilinos packages, which set DEFINES in Alany build ###

# MPI check -- defaults to TRUE
# MESSAGE("-- Checking if MPI is enabled in Trilinos:")
LIST(FIND Trilinos_TPL_LIST MPI MPI_List_ID)
IF (MPI_List_ID GREATER -1)
  MESSAGE("-- Looking for MPI in Trilinos:    Found, compiling with -DALBANY_MPI")
  SET(ALBANY_MPI TRUE)
  ADD_DEFINITIONS(-DALBANY_MPI)
ELSE()
  MESSAGE("-- Looking for MPI in Trilinos:    NOT found, MPI NOT ENABLED!")
  SET(ALBANY_MPI FALSE)
ENDIF()


# MESSAGE("-- Looking for Dakota (via TriKota):")
LIST(FIND Trilinos_PACKAGE_LIST TriKota TriKota_List_ID)
IF (TriKota_List_ID GREATER -1)
  ADD_DEFINITIONS(-DHAVE_CONFIG_H -DALBANY_DAKOTA)
  MESSAGE("-- Looking for Dakota via TriKota: Found, compiling with -DALBANY_DAKOTA")
  SET(ALBANY_DAKOTA TRUE)
ELSE()
  MESSAGE("-- Looking for Dakota via TriKota: NOT found.")
  SET(ALBANY_DAKOTA FALSE)
ENDIF()

# set optional dependency on SEACAS, for I/O via stk_io
# MESSAGE("-- Looking for SEACAS:")
LIST(FIND Trilinos_PACKAGE_LIST SEACASIoss SEACASIoss_List_ID)
IF (SEACASIoss_List_ID GREATER -1)
  ADD_DEFINITIONS(-DALBANY_SEACAS)
  MESSAGE("-- Looking for SEACASIoss:         Found, compiling with -DALBANY_SEACAS")
  SET(ALBANY_SEACAS TRUE)
ELSE()
  MESSAGE("-- Looking for SEACASIoss:         NOT found.")
  SET(ALBANY_SEACAS FALSE)

# Warning message for 8/15/11 trilinos change -- remove after dust settles
#  LIST(FIND Trilinos_PACKAGE_LIST SEACAS SEACAS_List_ID)
#  IF (SEACAS_List_ID GREATER -1)
#     MESSAGE("TRILINOS HAS CHANGED! Configure with Trilinos_ENABLE_SEACASIoss:BOOL=ON")
#  ENDIF()

ENDIF()

# set optional dependency on stk_percept, for uniform adaptivity
LIST(FIND Trilinos_LIBRARIES stk_percept stk_percept_List_ID)
IF (stk_percept_List_ID GREATER -1)
  ADD_DEFINITIONS(-DALBANY_STK_PERCEPT)
  MESSAGE("-- Looking for stk_percept:        Found, compiling with -DALBANY_STK_PERCEPT")
  SET(ALBANY_STK_PERCEPT TRUE)
ELSE()
  MESSAGE("-- Looking for stk_percept:        NOT found.")
  SET(ALBANY_STK_PERCEPT FALSE)
ENDIF()

# set optional dependency on Zoltan, for partitioning
# MESSAGE("-- Looking for Zoltan:")
LIST(FIND Trilinos_PACKAGE_LIST Zoltan Zoltan_List_ID)
IF (Zoltan_List_ID GREATER -1)
  ADD_DEFINITIONS(-DALBANY_ZOLTAN)
  MESSAGE("-- Looking for Zoltan:             Found, compiling with -DALBANY_ZOLTAN")
  SET(ALBANY_ZOLTAN TRUE)
ELSE()
  MESSAGE("-- Looking for Zoltan:             NOT found.")
  SET(ALBANY_ZOLTAN FALSE)
ENDIF()

# set optional dependency on RBGen, for reduced basis computation
LIST(FIND Trilinos_PACKAGE_LIST RBGen RBGen_List_ID)
IF (RBGen_List_ID GREATER -1)
  ADD_DEFINITIONS(-DALBANY_RBGEN)
  MESSAGE("-- Looking for RBGen:              Found, compiling with -DALBANY_RBGEN")
  SET(ALBANY_RBGEN TRUE)
ELSE()
  MESSAGE("-- Looking for RBGen:              NOT found.")
  SET(ALBANY_RBGEN FALSE)
ENDIF()

# Check for optional physics sets in Albany
MESSAGE("\nChecking which Albany physics sets are enabled:")

# set optional dependency on demoPDEs, defaults to Enabled
OPTION(ENABLE_DEMO_PDES "Flag to turn on demonstration PDEs problems" ON)
IF (ENABLE_DEMO_PDES)
  ADD_DEFINITIONS(-DALBANY_DEMO_PDES)
  MESSAGE("-- DEMO_PDES is Enabled, compiling with -DALBANY_DEMO_PDES")
  SET(ALBANY_DEMO_PDES TRUE)
ELSE()
  MESSAGE("-- DEMO_PDES is NOT Enabled.")
  SET(ALBANY_DEMO_PDES FALSE)
ENDIF()

# set optional dependency on LCM, defaults to Disabled
OPTION(ENABLE_LCM "Flag to turn on LCM Mechanics source code" OFF)
IF (ENABLE_LCM)
  ADD_DEFINITIONS(-DALBANY_LCM)
  MESSAGE("-- LCM       is Enabled, compiling with -DALBANY_LCM")
  SET(ALBANY_LCM TRUE)
ELSE()
  MESSAGE("-- LCM       is NOT Enabled.")
  SET(ALBANY_LCM FALSE)
ENDIF()

# set optional dependency on Peridigm peridynamics code, defaults to Disabled
OPTION(ENABLE_PERIDIGM "Flag to turn on Albany-Peridigm code coupling" OFF)
IF(ENABLE_PERIDIGM)
  SET(CMAKE_PREFIX_PATH ${PERIDIGM_DIR} ${CMAKE_PREFIX_PATH})
  FIND_PACKAGE(Peridigm REQUIRED)
  MESSAGE("\nFound Peridigm!  Here are the details: ")
  MESSAGE("   Peridigm_DIR = ${Peridigm_DIR}")
  MESSAGE("   Peridigm_LIBRARIES = ${Peridigm_LIBRARIES}")
  MESSAGE("   Peridigm_INCLUDE_DIRS = ${Peridigm_INCLUDE_DIRS}")
  MESSAGE("   Peridigm_LIBRARY_DIRS = ${Peridigm_LIBRARY_DIRS}")
  MESSAGE("End of Peridigm details\n")
  ADD_DEFINITIONS(-DALBANY_PERIDIGM)
  SET(ALBANY_PERIDIGM TRUE)
ELSE()
  MESSAGE("-- PERIDIGM  is NOT Enabled.")
  SET(ALBANY_PERIDIGM FALSE)
ENDIF()

# Set optional build of QCAD (Quantum Device Modeling and Design) defaults to ON
OPTION(ENABLE_QCAD "Flag to turn on QCAD Source code" ON)
IF (ENABLE_QCAD)
  ADD_DEFINITIONS(-DALBANY_QCAD)
  MESSAGE("-- QCAD      is Enabled, compiling with -DALBANY_QCAD")
  SET(ALBANY_QCAD TRUE)
ELSE()
  MESSAGE("-- QCAD      is NOT Enabled.")
  SET(ALBANY_QCAD FALSE)
ENDIF()

# Set optional build of FELIX (Land Ice SciDAC project), defaults to Disabled
OPTION(ENABLE_FELIX "Flag to turn on FELIX Source code" OFF)
IF (ENABLE_FELIX)
  ADD_DEFINITIONS(-DALBANY_FELIX)
  MESSAGE("-- FELIX     is Enabled, compiling with -DALBANY_FELIX")
  SET(ALBANY_FELIX TRUE)
ELSE()
  MESSAGE("-- FELIX     is NOT Enabled.")
  SET(ALBANY_FELIX FALSE)
ENDIF()

# Set optional build of CISM Interface for FELIX (Land Ice SciDAC project),
# defaults to Disabled

OPTION(ENABLE_CISM_INTERFACE "Flag to turn on FELIX Source code" OFF)
IF (ENABLE_CISM_INTERFACE AND ENABLE_FELIX)
  ADD_DEFINITIONS(-DCISM_HAS_FELIX)
  MESSAGE("-- CISM_INTERFACE     is Enabled, compiling with -DCISM_HAS_FELIX")
  SET(CISM_HAS_FELIX TRUE)
  IF (NOT CISM_INCLUDE_DIR)
    MESSAGE("\nError:  CISM_INCLUDE_DIR not defined\n")
  ELSE()
    MESSAGE("-- CISM_INCLUDE_DIR ${CISM_INCLUDE_DIR}")
  ENDIF()
ELSEIF(ENABLE_CISM_INTERFACE AND (NOT ENABLE_FELIX))
  MESSAGE(FATAL_ERROR "\nError:  ENABLE_CISM_INTERFACE requires ENABLE_FELIX
to be ON\n")
ELSE()
  MESSAGE("-- CISM_INTERFACE     is NOT Enabled.")
  SET(CISM_HAS_FELIX FALSE)
ENDIF()

# Set optional build of Aeras (Atmosphere Dynamics LDRD), defaults to Disabled
OPTION(ENABLE_AERAS "Flag to turn on Aeras Source code" OFF)
IF (ENABLE_AERAS)
  ADD_DEFINITIONS(-DALBANY_AERAS)
  MESSAGE("-- Aeras     is Enabled, compiling with -DALBANY_AERAS")
  SET(ALBANY_AERAS TRUE)
ELSE()
  MESSAGE("-- Aeras     is NOT Enabled.")
  SET(ALBANY_AERAS FALSE)
ENDIF()

# optional dependency on LAME, defaults to FALSE
OPTION(ENABLE_LAME "Flag to turn on LAME material library source code" OFF)
IF(ENABLE_LAME)
  MESSAGE("-- LAME      is enabled, compiling with -DALBANY_LAME")
  ADD_DEFINITIONS(-DALBANY_LAME)
  SET(ALBANY_LAME TRUE)
  IF(NOT LAME_INCLUDE_DIR)
    MESSAGE("\nError:  LAME_INCLUDE_DIR not defined\n")
  ELSE()
    MESSAGE("-- LAME_INCLUDE_DIR ${LAME_INCLUDE_DIR}")
  ENDIF()
  IF(NOT LAME_LIBRARY_DIR)
    MESSAGE("\nError:  LAME_LIBRARY_DIR not defined\n")
  ELSE()
    MESSAGE("-- LAME_LIBRARY_DIR ${LAME_LIBRARY_DIR}")
  ENDIF()
ELSE()
  MESSAGE("-- LAME      is NOT Enabled.")
  SET(ALBANY_LAME FALSE)
ENDIF()

# optional dependency on LAMENT, defaults to FALSE
OPTION(ENABLE_LAMENT "Flag to turn on LAMENT lmaterial library source code" OFF)
IF(ENABLE_LAMENT)
  MESSAGE("-- LAMENT    is enabled, compiling with -DALBANY_LAMENT")
  ADD_DEFINITIONS(-DALBANY_LAMENT)
  SET(ALBANY_LAMENT TRUE)
  IF(NOT LAMENT_INCLUDE_DIR)
    MESSAGE("\nError:  LAMENT_INCLUDE_DIR not defined\n")
  ELSE()
    MESSAGE("-- LAMENT_INCLUDE_DIR ${LAMENT_INCLUDE_DIR}")
  ENDIF()
  IF(NOT LAMENT_LIBRARY_DIR)
    MESSAGE("\nError:  LAMENT_LIBRARY_DIR not defined\n")
  ELSE()
    MESSAGE("-- LAMENT_LIBRARY_DIR ${LAMENT_LIBRARY_DIR}")
  ENDIF()
ELSE()
  MESSAGE("-- LAMENT    is NOT Enabled.")
  SET(ALBANY_LAMENT FALSE)
ENDIF()

# Add option for ASCR work, defaults to disabled
OPTION(ENABLE_ASCR "Flag to turn on ASCR-funded coupled UQ research source code" OFF)
IF (ENABLE_ASCR)
  ADD_DEFINITIONS(-DALBANY_ASCR)
  MESSAGE("-- ASCR      is Enabled, compiling with -DALBANY_ASCR")
  SET(ALBANY_ASCR TRUE)
ELSE()
  MESSAGE("-- ASCR      is NOT Enabled.")
  SET(ALBANY_ASCR FALSE)
ENDIF()

# Add option for Hydride work, defaults to disabled
OPTION(ENABLE_HYDRIDE "Flag to turn on Hydride source code" OFF)
IF (ENABLE_HYDRIDE)
  IF (NOT ENABLE_LCM)
    ADD_DEFINITIONS(-DALBANY_LCM)
    MESSAGE("-- HYDRIDE  requires LCM... compiling with -DALBANY_LCM")
    SET(ALBANY_LCM TRUE)
  ENDIF()
  ADD_DEFINITIONS(-DALBANY_HYDRIDE)
  MESSAGE("-- HYDRIDE   is Enabled, compiling with -DALBANY_HYDRIDE")
  SET(ALBANY_HYDRIDE TRUE)
ELSE()
  MESSAGE("-- HYDRIDE   is NOT Enabled.")
  SET(ALBANY_HYDRIDE FALSE)
ENDIF()

# Set optional build of MOR (Model Order Reduction) defaults to ON
OPTION(ENABLE_MOR "Flag to turn on MOR Source code" ON)
IF (ENABLE_MOR)
  ADD_DEFINITIONS(-DALBANY_MOR)
  MESSAGE("-- MOR       is Enabled, compiling with -DALBANY_MOR")
  SET(ALBANY_MOR TRUE)
ELSE()
  MESSAGE("-- MOR       is NOT Enabled.")
  SET(ALBANY_MOR FALSE)
ENDIF()

# Set optional build of ALBANY_CI Configuration Interaction, defaults to Disabled
OPTION(ENABLE_ALBANY_CI "Flag to turn on Configuration Interaction" OFF)
IF (ENABLE_ALBANY_CI)
  IF(NOT ENABLE_QCAD)
    MESSAGE(FATAL_ERROR "\nError:  ALBANY_CI requires ENABLE_QCAD to be ON\n")
  ENDIF()
  ADD_DEFINITIONS(-DALBANY_CI)
  SET(ALBANY_CI TRUE)
  MESSAGE("-- AlbanyCI  is Enabled, compiling with -DALBANY_CI")

  IF(NOT ALBANY_CI_INCLUDE_DIR)
    MESSAGE("\nError:  ALBANY_CI_INCLUDE_DIR not defined\n")
  ELSE()
    MESSAGE("-- ALBANY_CI_INCLUDE_DIR ${ALBANY_CI_INCLUDE_DIR}")
  ENDIF()
  IF(NOT ALBANY_CI_LIBRARY_DIR)
    MESSAGE("\nError:  ALBANY_CI_LIBRARY_DIR not defined\n")
  ELSE()
    MESSAGE("-- ALBANY_CI_LIBRARY_DIR ${ALBANY_CI_LIBRARY_DIR}")
  ENDIF()
ELSE()
  MESSAGE("-- AlbanyCI  is NOT Enabled.")
  SET(ALBANY_CI FALSE)
ENDIF()

# Set optional dependency on SCOREC FMDB,  optional adaptivity
# MESSAGE("-- Looking for SCOREC:")
LIST(FIND Trilinos_PACKAGE_LIST SCOREC SCOREC_List_ID)
IF(ENABLE_SCOREC AND SCOREC_List_ID GREATER -1)
#IF(SCOREC_List_ID GREATER -1)
  IF(NOT ALBANY_ZOLTAN)
    MESSAGE(FATAL_ERROR "\nError:  SCOREC option requires ZOLTAN with PARMETIS built in\n")
  ENDIF()
  # ParMETIS TPL check
  MESSAGE("-- Checking if ParMETIS is compiled into in Trilinos:")
  LIST(FIND Trilinos_TPL_LIST ParMETIS PARMETIS_List_ID)
  IF (PARMETIS_List_ID GREATER -1)
    MESSAGE("----> ParMETIS is a Trilinos TPL")
  ELSE()
    MESSAGE(FATAL_ERROR "\nError:  SCOREC option requires ZOLTAN with PARMETIS built in\n")
  ENDIF()
  MESSAGE("-- SCOREC    is enabled, compiling with -DALBANY_SCOREC")
  ADD_DEFINITIONS(-DALBANY_SCOREC -DTRILINOS)
  SET(ALBANY_SCOREC TRUE)
  IF(ALBANY_MPI)
    ADD_DEFINITIONS(-DPUMI_PARALLEL)
  ENDIF()
  LIST(FIND Trilinos_TPL_LIST Parasolid Parasolid_List_ID)
  IF (Parasolid_List_ID GREATER -1)
    MESSAGE("----> Parsolid is a Trilinos TPL")
    ADD_DEFINITIONS(-DSCOREC_PARASOLID)
    SET(ENABLE_PARASOLID TRUE)
  ENDIF()
  LIST(FIND Trilinos_TPL_LIST ACIS ACIS_List_ID)
  IF (ACIS_List_ID GREATER -1)
    MESSAGE("----> ACIS is a Trilinos TPL")
    ADD_DEFINITIONS(-DSCOREC_ACIS)
    SET(ENABLE_ACIS TRUE)
  ENDIF()
  IF (EXISTS ${Trilinos_LIB_DIRS}/libpumi_geom_meshmodel.a)
    MESSAGE("----> Mesh Model is enabled")
    ADD_DEFINITIONS(-DSCOREC_MESHMODEL)
    SET(ENABLE_MESHMODEL TRUE)
  ENDIF()
  IF (EXISTS ${Trilinos_LIB_DIRS}/libspr.a)
    MESSAGE("----> SPR is enabled")
    ADD_DEFINITIONS(-DSCOREC_SPR)
    SET(ENABLE_SPR TRUE)
  ENDIF()
ELSE()
  MESSAGE("-- SCOREC    is NOT Enabled.")
  SET(ALBANY_SCOREC FALSE)
ENDIF()

# ParaView's Catalyst CoProcessing package
OPTION(ENABLE_CATALYST "Flag to turn on support for the Catalyst CoProcessing library" OFF)
IF (ENABLE_CATALYST)
  ADD_DEFINITIONS(-DALBANY_CATALYST)
  MESSAGE("-- Catalyst  is Enabled, compiling with -DALBANY_CATALYST")
  SET(ALBANY_CATALYST TRUE)
ELSE()
  MESSAGE("-- Catalyst  is NOT Enabled.")
  SET(ALBANY_CATALYST FALSE)
ENDIF()

# Set optional build of StochasticGalerkin and MultiPoint types
# These are required for embedded UQ, but slow compilation considerably
OPTION(ENABLE_SG_MP "Flag to turn on SG_MP StochasticGalerkin and MP Code" ON)
IF (ENABLE_SG_MP)
  ADD_DEFINITIONS(-DALBANY_SG_MP)
  MESSAGE("-- SG_MP     is Enabled, compiling with -DALBANY_SG_MP")
  SET(ALBANY_SG_MP TRUE)
ELSE()
  MESSAGE("-- SG_MP     is NOT Enabled.")
  SET(ALBANY_SG_MP FALSE)
  IF(ENABLE_ASCR)
    MESSAGE(FATAL_ERROR "\nError:  ASCR code requires ENABLE_SG_MP to be ON\n")
  ENDIF()
ENDIF()

## Other Miscellaneous Albany Options here:
MESSAGE("\nMiscellaneous Albany Options:")

# Set optional build of FELIX (Land Ice SciDAC project), defaults to Disabled
OPTION(ENABLE_FAST_FELIX "Flag to turn on Code Optimization for FELIX that
may break other physics" OFF)
IF (ENABLE_FELIX AND ENABLE_FAST_FELIX)
  ADD_DEFINITIONS(-DALBANY_FAST_FELIX)
  MESSAGE("-- FAST_FELIX               is Enabled: MIGHT BREAK OTHER PHYSICS!!")
ENDIF()


# Add option to check for hardware floating point exceptions using SSE, defaults to disabled
OPTION(ENABLE_CHECK_FPE "Flag to turn on floating point exception handling" OFF)
IF (ENABLE_CHECK_FPE)
  ADD_DEFINITIONS(-DALBANY_CHECK_FPE)
  MESSAGE("-- FPE checking              is Enabled, compiling with -DALBANY_CHECK_FPE")
  SET(ALBANY_CHECK_FPE TRUE)
ELSE()
  MESSAGE("-- FPE checking              is NOT Enabled.")
  SET(ALBANY_CHECK_FPE FALSE)
ENDIF()

OPTION(ENABLE_ALBANY_VERBOSE "Flag to turn on verbose output" OFF)
IF (NOT ENABLE_ALBANY_VERBOSE)
  MESSAGE("-- Verbose printing          is NOT Enabled")
  SET(ALBANY_VERBOSE FALSE)
ELSE()
  ADD_DEFINITIONS(-DALBANY_VERBOSE)
  MESSAGE("-- Verbose printing          is Enabled")
  SET(ALBANY_VERBOSE TRUE)
ENDIF()

# Option to enable "speculative" features that are not ready for general visibility
OPTION(ENABLE_LCM_SPECULATIVE "Flag to turn on speculative LCM features/code" OFF)
IF (NOT ENABLE_LCM_SPECULATIVE)
  MESSAGE("-- Speculative LCM features  are NOT Enabled")
  SET(LCM_SPECULATIVE FALSE)
ELSE()
  ADD_DEFINITIONS(-DLCM_SPECULATIVE)
  MESSAGE("-- Speculative LCM features  are Enabled")
  SET(LCM_SPECULATIVE TRUE)
ENDIF()

# Enhanced debugging option
OPTION(ENABLE_DEBUGGING "Flag to turn on enhanced debugging info" OFF)
IF (ENABLE_DEBUGGING)
  ADD_DEFINITIONS(-DALBANY_DEBUG)
  MESSAGE("-- Enhanced debugging        is Enabled, compiling with -DALBANY_DEBUG")
  SET(ALBANY_DEBUG TRUE)
ELSE()
  MESSAGE("-- Enhanced debugging        is NOT Enabled.")
  SET(ALBANY_DEBUG FALSE)
ENDIF()

# Add option to enable performance tests. Results are machine-specific
OPTION(ENABLE_PERFORMANCE_TESTS "Flag to turn on Performance Tests" OFF)
IF (ENABLE_PERFORMANCE_TESTS)
  ADD_DEFINITIONS(-DALBANY_PERFORMANCE_TESTS)
  MESSAGE("-- Performance Tests         are Enabled, adding examples/PerformanceTests")
  SET(ALBANY_PERFORMANCE_TESTS TRUE)
ELSE()
  MESSAGE("-- Performance Tests         are NOT Enabled.")
  SET(ALBANY_PERFORMANCE_TESTS FALSE)
ENDIF()

# Add option to include Dakota restart examples in the test suite
IF (NOT APPLE)
CMAKE_DEPENDENT_OPTION(ENABLE_DAKOTA_RESTART_EXAMPLES "Flag to include Dakota restart examples" ON
                       "ALBANY_DAKOTA" OFF)
ELSE (NOT APPLE)
# Portable Dakota restart files is broken on MacOS, deactivate test by default
CMAKE_DEPENDENT_OPTION(ENABLE_DAKOTA_RESTART_EXAMPLES "Flag to include Dakota restart examples" OFF
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

# add a target to generate API documentation with Doxygen
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

## End of CMake options to Albany

add_subdirectory(src)

enable_testing()

#  Timeout real long running tests

IF(ALBANY_CTEST_TIMEOUT)
    MESSAGE("-- Setting maximum test run time to be       ${ALBANY_CTEST_TIMEOUT} seconds")
ELSE()
    MESSAGE("-- No limits are set on maximum test run time")
ENDIF()

# Make a clean build target for convenience

ADD_CUSTOM_TARGET( cmake_clean
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/Makefile
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/cmake_install.cmake
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake
)

IF(ALBANY_MPI)
  IF(ALBANY_MPI_OPTIONS)
    SET(MPIEX  ${ALBANY_MPI_EXEC})
    SET(MPINPF ${ALBANY_MPI_EXEC_NUMPROCS_FLAG})
    SET(MPIMNP ${ALBANY_MPI_EXEC_MAX_NUMPROCS})
    SET(MPIPRE ${ALBANY_MPI_LEADING_OPTIONS})
    # For now - assume ALBANY_MPI_OPTIONS means to ONLY attempt the parallel
    # tests
    SET(ALBANY_PARALLEL_ONLY true)
    MESSAGE("-- CTest mpi_exec call:  ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP}")
  ELSE()
    SET(MPIEX  ${Trilinos_MPI_EXEC})
    SET(MPINPF ${Trilinos_MPI_EXEC_NUMPROCS_FLAG})
    SET(MPIMNP ${Trilinos_MPI_EXEC_MAX_NUMPROCS})
    SET(MPIPRE "")
    SET(ALBANY_PARALLEL_ONLY false)
    MESSAGE("-- CTest mpi_exec call:  ${MPIEX} ${MPINPF} ${MPIMNP}")
  ENDIF()
ENDIF(ALBANY_MPI)

add_subdirectory(examples)
MESSAGE("\nAnd CMake says...:")
