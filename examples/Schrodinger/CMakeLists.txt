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
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input1D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input1D.xml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and executable
add_test(${testName}_1D ${Albany.exe} input1D.xml)

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input2D.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square2D.exo
               ${CMAKE_CURRENT_BINARY_DIR}/square2D.exo COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and executable
add_test(${testName}_2D_SERIAL ${SerialAlbany.exe} input2D.xml)

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input3D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input3D.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube3D.exo
               ${CMAKE_CURRENT_BINARY_DIR}/cube3D.exo COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and executable
add_test(${testName}_3D_SERIAL ${SerialAlbany.exe} input3D.xml)


