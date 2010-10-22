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
FIND_PACKAGE(Trilinos PATHS ${ALBANY_TRILINOS_DIR}/include)

IF(NOT Trilinos_FOUND)
  MESSAGE(FATAL_ERROR "Could not find Trilinos!")
ENDIF()

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
MESSAGE("End of Trilinos details\n")

SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER} )
SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER} )
SET(CMAKE_Fortran_COMPILER ${Trilinos_Fortran_COMPILER} )

IF (CMAKE_Fortran_COMPILER) # Enable Fortran if it is enabled in Trilinos.
  ENABLE_LANGUAGE(Fortran) 
ENDIF()

add_definitions("-g -O2")
# End of setup and error checking
#  NOTE: PROJECT command checks for compilers, so this statement
#        is moved AFTER setting CMAKE_CXX_COMPILER from Trilinos

PROJECT(Albany)


MESSAGE("-- Looking for Dakota (via TriKota):")
LIST(FIND Trilinos_PACKAGE_LIST TriKota TriKota_List_ID)
IF (TriKota_List_ID GREATER -1)
  ADD_DEFINITIONS(-DHAVE_CONFIG_H -DALBANY_DAKOTA)
  MESSAGE("-- Looking for Dakota (via TriKota): -- found, compiling with -DHAVE_DAKOTA")
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

# set optional dependency on Ioss, for I/O via stk_io
MESSAGE("-- Looking for Ioss:")
LIST(FIND Trilinos_PACKAGE_LIST Trios Trios_List_ID)
IF (Trios_List_ID GREATER -1)
  ADD_DEFINITIONS(-DALBANY_IOSS)
  MESSAGE("-- Looking for Ioss: -- found, compiling with -DALBANY_IOSS")
  SET(ALBANY_IOSS TRUE)
ELSE()
  MESSAGE("-- Looking for Ioss: -- not found.")
  SET(ALBANY_IOSS FALSE)
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
