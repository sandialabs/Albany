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

# Copy files used by multiple tests
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)

# Test name root is the directory name
get_filename_component(testRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_test2D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_test2D.xml COPYONLY)
# 3. Create the test with this name and standard executable
add_test(${testRoot}_test2D_SERIAL ${SerialAlbany.exe} input_test2D.xml)


# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_pmoscap.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_pmoscap.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pmoscap.exo
               ${CMAKE_CURRENT_BINARY_DIR}/pmoscap.exo COPYONLY)
# 3. Create the test with this name and standard executable
add_test(${testRoot}_pmoscap_SERIAL ${SerialAlbany.exe} input_pmoscap.xml)


# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_pndiode.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_pndiode.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pndiode.exo
               ${CMAKE_CURRENT_BINARY_DIR}/pndiode.exo COPYONLY)
# 3. Create the test with this name and standard executable
add_test(${testRoot}_pndiode_SERIAL ${SerialAlbany.exe} input_pndiode.xml)


# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_mosdot_2D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_mosdot_2D.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mosdot_2D.exo
               ${CMAKE_CURRENT_BINARY_DIR}/mosdot_2D.exo COPYONLY)
# 3. Create the test with this name and standard executable
add_test(${testRoot}_mosdot_2D_SERIAL ${SerialAlbany.exe} input_mosdot_2D.xml)

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_mosdot_3D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_mosdot_3D.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mosdot_3D.exo
               ${CMAKE_CURRENT_BINARY_DIR}/mosdot_3D.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mosdot_3D.exo.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/mosdot_3D.exo.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mosdot_3D.exo.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/mosdot_3D.exo.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mosdot_3D.exo.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/mosdot_3D.exo.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mosdot_3D.exo.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/mosdot_3D.exo.4.3 COPYONLY)
# 3. Create the test with this name and standard executable
add_test(${testRoot}_mosdot_3D ${Albany.exe} input_mosdot_3D.xml)
