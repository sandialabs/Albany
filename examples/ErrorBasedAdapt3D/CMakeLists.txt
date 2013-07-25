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
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input-elasticity.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input-elasticity.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input-plasticity.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input-plasticity.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials-elasticity.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials-elasticity.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials-plasticity.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials-plasticity.xml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/bar_with_hole_FMDB.sms
               ${CMAKE_CURRENT_BINARY_DIR}/bar_with_hole_FMDB.sms COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coarse_bar_with_hole_FMDB.sms
               ${CMAKE_CURRENT_BINARY_DIR}/coarse_bar_with_hole_FMDB.sms COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/very_coarse_FMDB.sms
               ${CMAKE_CURRENT_BINARY_DIR}/very_coarse_FMDB.sms COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/bar_with_hole.xmt_txt
               ${CMAKE_CURRENT_BINARY_DIR}/bar_with_hole.xmt_txt COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable

#add_test(${testName}_Elastic ${Albany.exe} input-elasticity.xml)
#add_test(${testName}_Plastic ${Albany.exe} input-plasticity.xml)
add_test(${testName}_Elastic ${Albany.exe} inputElastML.xml)
add_test(${testName}_Plastic ${Albany.exe} inputCoarseML.xml)

