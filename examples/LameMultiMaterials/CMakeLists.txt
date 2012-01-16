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
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSingleMaterial.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSingleMaterial.xml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/10_blocks_1.0_size_1.0_meshSize.g
               ${CMAKE_CURRENT_BINARY_DIR}/10_blocks_1.0_size_1.0_meshSize.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMultiMat_10.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMultiMat_10.xml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_10.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials_10.xml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/100_blocks_1.0_size_1.0_meshSize.g
               ${CMAKE_CURRENT_BINARY_DIR}/100_blocks_1.0_size_1.0_meshSize.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMultiMat_100.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMultiMat_100.xml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_100.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials_100.xml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
add_test(${testName}_SingleMaterial_SERIAL ${SerialAlbany.exe} inputSingleMaterial.xml)

add_test(${testName}_MultiMaterial_10_SERIAL ${SerialAlbany.exe} inputMultiMat_10.xml)
add_test(${testName}_MultiMaterial_100_SERIAL ${SerialAlbany.exe} inputMultiMat_100.xml)

