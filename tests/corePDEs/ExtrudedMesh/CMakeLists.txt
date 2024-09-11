if (ALBANY_IFPACK2 AND ALBANY_OMEGAH)
  # Name the test with the directory name
  get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
  get_filename_component(parentDirName ${parentPath} NAME)
  get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  set (testNameRoot ${parentDirName}_${dirName})

  # Create the test with this name and standard executable
  # Note: only extruded.yaml is used, but we ship also monolithic,
  #       which uses a STK3D mesh, so you can easily debug by
  #       running both the yamls.
  #       PLEASE, keep them in sync!
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/extruded.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/extruded.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/monolithic.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/monolithic.yaml COPYONLY)
  add_test(${testNameRoot} ${Albany.exe} extruded.yaml)
  set_tests_properties(${testNameRoot} PROPERTIES LABELS "Basic;Forward")
endif ()
