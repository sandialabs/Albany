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
#* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *hole-m4-bKL.par.4.1hole-m4-bKL.par.4.1
#* including this sentence must appear on any copies of this software.*
#*    Questions to Glen Hansen, gahanse@sandia.gov                    *
#\********************************************************************/


# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)

# 2. Copy mesh files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hole.exo
               ${CMAKE_CURRENT_BINARY_DIR}/hole.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hole.exo.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/hole.exo.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hole.exo.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/hole.exo.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hole.exo.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/hole.exo.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hole.exo.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/hole.exo.4.3 COPYONLY)

# 3. Copy runtest.cmake from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake COPYONLY)

# 4. Build the test executable to read the ncdump file
add_executable(TestNcdumpValues TestNcdumpValues.C)

# 5. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 6. Run the "runtest.cmake" file in the CWD

add_test(NAME ${testName} COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
  -DTEST_NAME=${testName} -DMPIMNP=${MPIMNP} -P runtest.cmake)
