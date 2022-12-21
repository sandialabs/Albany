IF(NOT ALBANY_PARALLEL_ONLY)
  # Name the test with the directory name
  get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
  get_filename_component(parentDirName ${parentPath} NAME)
  get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  set (testNameRoot ${parentDirName}_${dirName})

  # Copy mesh file from source dir to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.exo
                 ${CMAKE_CURRENT_BINARY_DIR}/cube.exo COPYONLY)

  # Create the test with this name and standard executable
  if (ALBANY_EPETRA) 
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
    add_test(${testNameRoot}_Epetra ${SerialAlbany.exe} input.yaml)
    set_tests_properties(${testNameRoot}_Epetra PROPERTIES LABELS "Basic;Epetra;Forward")
  endif()
  if (ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)
    add_test(${testNameRoot}_Tpetra ${SerialAlbany.exe} inputT.yaml)
    set_tests_properties(${testNameRoot}_Tpetra PROPERTIES LABELS "Basic;Tpetra;Forward")
  endif ()
ENDIF()
