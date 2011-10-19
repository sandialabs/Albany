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
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_heat.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_heat.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_neutronics.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_neutronics.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_domain_coupled.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_domain_coupled.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_network_coupled.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_network_coupled.xml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testNameDomain ${CMAKE_CURRENT_SOURCE_DIR}_Domain NAME)
get_filename_component(testNameNetwork ${CMAKE_CURRENT_SOURCE_DIR}_Network NAME)
# 3. Create the test with this name and standard executable
add_test(${testNameDomain} ${Albany.exe} input_domain_coupled.xml)
add_test(${testNameNetwork} ${AlbanyCoupled.exe} input_heat.xml
input_neutronics.xml input_network_coupled.xml)

# 4. Repeat process for SG problems
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_heat_sg.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_heat_sg.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_neutronics_sg.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_neutronics_sg.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_domain_coupled_sg.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_domain_coupled_sg.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_network_coupled_sg.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_network_coupled_sg.xml COPYONLY)
get_filename_component(testNameDomainSG ${CMAKE_CURRENT_SOURCE_DIR}_Domain_SG NAME)
get_filename_component(testNameNetworkSG ${CMAKE_CURRENT_SOURCE_DIR}_Network_SG NAME)
add_test(${testNameDomainSG} ${AlbanySG.exe} input_domain_coupled_sg.xml)
add_test(${testNameNetworkSG} ${AlbanySGCoupled.exe} input_heat_sg.xml
input_neutronics_sg.xml input_network_coupled_sg.xml)
