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
# 2. Copy materials file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)

# 3. Copy mesh files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fuel_volume.exo
               ${CMAKE_CURRENT_BINARY_DIR}/fuel_volume.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fuel_volume.exo.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/fuel_volume.exo.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fuel_volume.exo.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/fuel_volume.exo.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fuel_volume.exo.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/fuel_volume.exo.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fuel_volume.exo.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/fuel_volume.exo.4.3 COPYONLY)

# 4. CylHeatProfile executable needs the ncdump file for mesh coordinates
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fuel_volume.ncdump
               ${CMAKE_CURRENT_BINARY_DIR}/fuel_volume.ncdump COPYONLY)

# 5. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 6. Create the test (actually the data for the test) with this name and standard executable

add_executable(CylHeatProfile CylHeatProfile.C)

add_test(${testName}_Prep CylHeatProfile)

# 7. Create the test with this name and standard executable

add_test(${testName} ${Albany.exe})
set_tests_properties(${testName}  PROPERTIES DEPENDS ${testName}_Prep)
