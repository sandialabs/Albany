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
#*    Questions to Glen Hansen, gahanse@sandia.gov                    *
#\********************************************************************/


# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_nobridge.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_nobridge.xml COPYONLY)

# 2. Copy mesh file from source to binary dir. We only need a single file as there is only
#    one master process used in this test.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ring.exo
               ${CMAKE_CURRENT_BINARY_DIR}/ring.exo COPYONLY)

# 3. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 4. Create the MPALE_dummy executable, for Albany to start up. The dummy executable just echos
#    back the stress values at the QPs.

add_executable(MPALE_dummy MPALE_dummy.C)

# 7. Run the Albany test

add_test(${testName}_nobridge_SERIAL ${SerialAlbany.exe} input_nobridge.xml)
