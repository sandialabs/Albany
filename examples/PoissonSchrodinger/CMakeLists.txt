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

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSchroPo.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSchroPo.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputCIPo.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputCIPo.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputPoissonMOSDot.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputPoissonMOSDot.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSchrodingerMOSDot.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSchrodingerMOSDot.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mosDotQuantum.exo
               ${CMAKE_CURRENT_BINARY_DIR}/mosDotQuantum.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mosDotQuantum.exo.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/mosDotQuantum.exo.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mosDotQuantum.exo.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/mosDotQuantum.exo.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mosDotQuantum.exo.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/mosDotQuantum.exo.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mosDotQuantum.exo.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/mosDotQuantum.exo.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)#

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
add_test(${testName}_SchroPo ${Albany.exe} inputSchroPo.xml)

IF (ALBANY_CI)
add_test(${testName}_CIPo ${Albany.exe} inputSchroPo.xml)
ENDIF()
