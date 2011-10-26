#/********************************************************************\
#*            Albany, Copyright (2010) Sandia Corporation             *
#*                                                                    *
#* Notice: This computer software was prepared by Sandia Corporation, *
#* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
#* the Department of Energy (DOE). All rights in the computer software*
#* are reserved by DOE on behalf of the United States Government and  *
#* the Contractor as provided in the Contract. You are authorized to  *
#* use this computer software for Governmental purposes but it is not *
#* to be released or distributed to the public. NEITHER THE GOVERNMENT*
#* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
#* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
#* including this sentence must appear on any copies of this software.*
#*    Questions to Andy Salinger, agsalin@sandia.gov                  *
#\********************************************************************/

# CMAKE File for Albany building against an installed Trilinos

cmake_minimum_required(VERSION 2.7)

# Error check up front
IF (NOT DEFINED ALBANY_TRILINOS_DIR)
  MESSAGE(FATAL_ERROR "\nAlbany Error: cmake must define ALBANY_TRILINOS_DIR:
  (-D ALBANY_TRILINOS_DIR:FILEPATH=<trilinos_install_prefix>)!")
ENDIF()

# Get Trilinos as one entity
SET(CMAKE_PREFIX_PATH ${ALBANY_TRILINOS_DIR} ${CMAKE_PREFIX_PATH})
FIND_PACKAGE(Trilinos REQUIRED)

MESSAGE("\nFound Trilinos!  Here are the details: ")
MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
MESSAGE("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
MESSAGE("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
MESSAGE("   Trilinos_LIBRARY_DIRS = ${Trilinos_LIBRARY_DIRS}")
MESSAGE("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
MESSAGE("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
MESSAGE("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
MESSAGE("   Trilinos_TPL_LIBRARY_DIRS = ${Trilinos_TPL_LIBRARY_DIRS}")
MESSAGE("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
MESSAGE("   Trilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
MESSAGE("End of Trilinos details\n")

SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER} )
SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER} )
SET(CMAKE_Fortran_COMPILER ${Trilinos_Fortran_COMPILER} )

IF (CMAKE_Fortran_COMPILER) # Enable Fortran if it is enabled in Trilinos.
  ENABLE_LANGUAGE(Fortran) 
ENDIF()

# End of setup and error checking
#  NOTE: PROJECT command checks for compilers, so this statement
#        is moved AFTER setting CMAKE_CXX_COMPILER from Trilinos

PROJECT(Albany)

IF (NOT CMAKE_CXX_FLAGS)
  SET(CMAKE_CXX_FLAGS ${Trilinos_CXX_COMPILER_FLAGS} )
ENDIF()
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ALBANY_CXX_FLAGS}")


MESSAGE("-- Looking for Dakota (via TriKota):")
LIST(FIND Trilinos_PACKAGE_LIST TriKota TriKota_List_ID)
IF (TriKota_List_ID GREATER -1)
  ADD_DEFINITIONS(-DHAVE_CONFIG_H -DALBANY_DAKOTA)
  MESSAGE("-- Looking for Dakota (via TriKota): -- found, compiling with -DALBANY_DAKOTA")
  SET(ALBANY_DAKOTA TRUE)
ELSE()
  MESSAGE("-- Looking for Dakota (via TriKota): -- not found.")
  SET(ALBANY_DAKOTA FALSE)
ENDIF()

# MPI check -- defaults to TRUE
MESSAGE("-- Checking if MPI is enabled in Trilinos:")
LIST(FIND Trilinos_TPL_LIST MPI MPI_List_ID)
IF (MPI_List_ID GREATER -1)
  MESSAGE("-- Checking if MPI is enabled in Trilinos: MPI ENABLED")
  SET(ALBANY_MPI TRUE)
  ADD_DEFINITIONS(-DALBANY_MPI)
ELSE()
  MESSAGE("-- Checking if MPI is enabled in Trilinos: MPI NOT ENABLED")
  SET(ALBANY_MPI FALSE)
ENDIF()

# set optional dependency on SEACAS, for I/O via stk_io
MESSAGE("-- Looking for SEACAS:")
LIST(FIND Trilinos_PACKAGE_LIST SEACASIoss SEACASIoss_List_ID)
IF (SEACASIoss_List_ID GREATER -1)
  ADD_DEFINITIONS(-DALBANY_SEACAS)
  MESSAGE("-- Looking for SEACASIoss: -- found, compiling with -DALBANY_SEACAS")
  SET(ALBANY_SEACAS TRUE)
ELSE()
  MESSAGE("-- Looking for SEACASIoss: -- not found.")
  SET(ALBANY_SEACAS FALSE)

# Warning message for 8/15/11 trilinos change -- remove after dust settles
  LIST(FIND Trilinos_PACKAGE_LIST SEACAS SEACAS_List_ID)
  IF (SEACAS_List_ID GREATER -1)
     MESSAGE("TRILINOS HAS CHANGED! Configure with Trilinos_ENABLE_SEACASIoss:BOOL=ON")
  ENDIF()

ENDIF()

# set optional dependency on Zoltan, for partitioning
MESSAGE("-- Looking for Zoltan:")
LIST(FIND Trilinos_PACKAGE_LIST Zoltan Zoltan_List_ID)
IF (Zoltan_List_ID GREATER -1)
  ADD_DEFINITIONS(-DALBANY_ZOLTAN)
  MESSAGE("-- Looking for Zoltan: -- found, compiling with -DALBANY_ZOLTAN")
  SET(ALBANY_ZOLTAN TRUE)
ELSE()
  MESSAGE("-- Looking for Zoltan: -- not found.")
  SET(ALBANY_ZOLTAN FALSE)
ENDIF()

# set optional dependency on LCM, defaults to Enabled
##    This changes default to ON:
##    IF (NOT ENABLE_LCM AND DEFINED ENABLE_LCM)
IF (NOT ENABLE_LCM)
  MESSAGE("-- LCM is NOT Enabled.")
  SET(ALBANY_LCM FALSE)
ELSE()
  ADD_DEFINITIONS(-DALBANY_LCM)
  MESSAGE("-- LCM is Enabled, compiling with -DALBANY_LCM")
  SET(ALBANY_LCM TRUE)
ENDIF()

# optional dependency on LAME, defaults to FALSE
IF(ENABLE_LAME)
  MESSAGE("-- LAME is enabled, compiling with -DALBANY_LAME")
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
  MESSAGE("-- LAME is NOT enabled.")
  SET(ALBANY_LAME FALSE)
ENDIF()

# Add option for ASCR work, defaults to disabled
IF (ENABLE_ASCR)
  ADD_DEFINITIONS(-DALBANY_ASCR)
  MESSAGE("-- ASCR is Enabled, compiling with -DALBANY_ASCR")
  SET(ALBANY_ASCR TRUE)
ELSE()
  MESSAGE("-- ASCR is NOT Enabled.")
  SET(ALBANY_ASCR FALSE)
ENDIF()

add_subdirectory(src)

enable_testing()
IF(ALBANY_MPI)
  SET(MPIEX  ${Trilinos_MPI_EXEC})
  SET(MPINPF ${Trilinos_MPI_EXEC_NUMPROCS_FLAG})
  SET(MPIMNP ${Trilinos_MPI_EXEC_MAX_NUMPROCS})
  MESSAGE("-- CTest mpi_exec call:  ${MPIEX} ${MPINPF} ${MPIMNP}")
ENDIF(ALBANY_MPI)

add_subdirectory(examples)
