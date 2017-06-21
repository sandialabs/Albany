# Change coupled tests to run on 4 proc's
IF (ALBANY_MPI)
  set(AlbanyCoupled.exe   ${MPIEX} ${MPIPRE} ${MPINPF} 4 ${Albany_BINARY_DIR}/src/AlbanyCoupled)
  set(AlbanySGCoupled.exe ${MPIEX} ${MPIPRE} ${MPINPF} 4 ${Albany_BINARY_DIR}/src/AlbanySGCoupled)
ENDIF()

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pipe.xml
               ${CMAKE_CURRENT_BINARY_DIR}/pipe.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pipe_coarse.exo.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/pipe_coarse.exo.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pipe_coarse.exo.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/pipe_coarse.exo.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pipe_coarse.exo.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/pipe_coarse.exo.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pipe_coarse.exo.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/pipe_coarse.exo.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reactor.xml
               ${CMAKE_CURRENT_BINARY_DIR}/reactor.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reactor_coarse.exo.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/reactor_coarse.exo.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reactor_coarse.exo.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/reactor_coarse.exo.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reactor_coarse.exo.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/reactor_coarse.exo.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reactor_coarse.exo.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/reactor_coarse.exo.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/network_coupled.xml
               ${CMAKE_CURRENT_BINARY_DIR}/network_coupled.xml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testNamePipe ${CMAKE_CURRENT_SOURCE_DIR}_Pipe NAME)
get_filename_component(testNameReactor ${CMAKE_CURRENT_SOURCE_DIR}_Reactor NAME)
get_filename_component(testNameNetwork ${CMAKE_CURRENT_SOURCE_DIR}_Network NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testNamePipe} ${Albany.exe} pipe.xml)
add_test(${testNameReactor} ${Albany.exe} reactor.xml)
add_test(${testNameNetwork} ${AlbanyCoupled.exe} network_coupled.xml)
endif()

# 4. Repeat process for SG problems
if(ALBANY_DAKOTA)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pipe_sg.xml
               ${CMAKE_CURRENT_BINARY_DIR}/pipe_sg.xml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reactor_sg.xml
               ${CMAKE_CURRENT_BINARY_DIR}/reactor_sg.xml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/network_coupled_sg.xml
               ${CMAKE_CURRENT_BINARY_DIR}/network_coupled_sg.xml COPYONLY)
  get_filename_component(testNamePipeSG ${CMAKE_CURRENT_SOURCE_DIR}_PipeSG NAME)
  get_filename_component(testNameReactorSG ${CMAKE_CURRENT_SOURCE_DIR}_ReactorSG NAME)
  get_filename_component(testNameNetworkSG ${CMAKE_CURRENT_SOURCE_DIR}_NetworkSG NAME)

  add_test(${testNamePipeSG} ${AlbanySG.exe} pipe_sg.xml)
  add_test(${testNameReactorSG} ${AlbanySG.exe} reactor_sg.xml)
  add_test(${testNameNetworkSG} ${AlbanySGCoupled.exe} network_coupled_sg.xml)
endif()
