# Change coupled tests to run on 4 proc's
IF (ALBANY_MPI)
  set(AlbanyCoupled.exe   ${MPIEX} ${MPIPRE} ${MPINPF} 4 ${Albany_BINARY_DIR}/src/AlbanyCoupled)
ENDIF()

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pipe.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/pipe.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pipe_coarse.exo.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/pipe_coarse.exo.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pipe_coarse.exo.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/pipe_coarse.exo.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pipe_coarse.exo.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/pipe_coarse.exo.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pipe_coarse.exo.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/pipe_coarse.exo.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reactor.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/reactor.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reactor_coarse.exo.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/reactor_coarse.exo.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reactor_coarse.exo.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/reactor_coarse.exo.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reactor_coarse.exo.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/reactor_coarse.exo.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reactor_coarse.exo.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/reactor_coarse.exo.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/network_coupled.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/network_coupled.yaml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testNamePipe ${CMAKE_CURRENT_SOURCE_DIR}_Pipe NAME)
get_filename_component(testNameReactor ${CMAKE_CURRENT_SOURCE_DIR}_Reactor NAME)
get_filename_component(testNameNetwork ${CMAKE_CURRENT_SOURCE_DIR}_Network NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testNamePipe} ${Albany.exe} pipe.yaml)
add_test(${testNameReactor} ${Albany.exe} reactor.yaml)
add_test(${testNameNetwork} ${AlbanyCoupled.exe} network_coupled.yaml)
endif()
