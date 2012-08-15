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
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pipe.xml
               ${CMAKE_CURRENT_BINARY_DIR}/pipe.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pipe_coarse.exo
               ${CMAKE_CURRENT_BINARY_DIR}/pipe_coarse.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reactor.xml
               ${CMAKE_CURRENT_BINARY_DIR}/reactor.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reactor_coarse.exo
               ${CMAKE_CURRENT_BINARY_DIR}/reactor_coarse.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/network_coupled.xml
               ${CMAKE_CURRENT_BINARY_DIR}/network_coupled.xml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testNamePipe ${CMAKE_CURRENT_SOURCE_DIR}_Pipe NAME)
get_filename_component(testNameReactor ${CMAKE_CURRENT_SOURCE_DIR}_Reactor NAME)
get_filename_component(testNameNetwork ${CMAKE_CURRENT_SOURCE_DIR}_Network NAME)
# 3. Create the test with this name and standard executable
add_test(${testNamePipe} ${SerialAlbany.exe} pipe.xml)
add_test(${testNameReactor} ${SerialAlbany.exe} reactor.xml)
add_test(${testNameNetwork} ${SerialAlbanyCoupled.exe} network_coupled.xml)

# 4. Repeat process for SG problems
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pipe_sg.xml
               ${CMAKE_CURRENT_BINARY_DIR}/pipe_sg.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reactor_sg.xml
               ${CMAKE_CURRENT_BINARY_DIR}/reactor_sg.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/network_coupled_sg.xml
               ${CMAKE_CURRENT_BINARY_DIR}/network_coupled_sg.xml COPYONLY)
get_filename_component(testNamePipeSG ${CMAKE_CURRENT_SOURCE_DIR}_PipeSG NAME)
get_filename_component(testNameReactorSG ${CMAKE_CURRENT_SOURCE_DIR}_ReactorSG NAME)
get_filename_component(testNameNetworkSG ${CMAKE_CURRENT_SOURCE_DIR}_NetworkSG NAME)
add_test(${testNamePipeSG} ${SerialAlbanySG.exe} pipe_sg.xml)
add_test(${testNameReactorSG} ${SerialAlbanySG.exe} reactor_sg.xml)
add_test(${testNameNetworkSG} ${SerialAlbanySGCoupled.exe} network_coupled_sg.xml)
