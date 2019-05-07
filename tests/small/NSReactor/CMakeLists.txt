if (ALBANY_EPETRA) 

  # Copy input and mesh files from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reactor.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/reactor.yaml COPYONLY)
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
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reactor_coarse.exo.4.0
                 ${CMAKE_CURRENT_BINARY_DIR}/reactor_coarse.exo.4.0 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reactor_coarse.exo.4.1
                 ${CMAKE_CURRENT_BINARY_DIR}/reactor_coarse.exo.4.1 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reactor_coarse.exo.4.2
                 ${CMAKE_CURRENT_BINARY_DIR}/reactor_coarse.exo.4.2 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reactor_coarse.exo.4.3
                 ${CMAKE_CURRENT_BINARY_DIR}/reactor_coarse.exo.4.3 COPYONLY)

  # Create the tests with this name and standard executable
  add_test(${testName}_Pipe_Epetra ${Albany.exe} pipe.yaml)
  set_tests_properties(${testName}_Pipe_Epetra PROPERTIES LABELS "Demo;Epetra;Forward")

  add_test(${testName}_Reactor_Epetra ${Albany.exe} reactor.yaml)
  set_tests_properties(${testName}_Reactor_Epetra PROPERTIES LABELS "Demo;Epetra;Forward")
endif()
