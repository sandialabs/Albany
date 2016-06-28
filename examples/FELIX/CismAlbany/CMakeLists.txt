add_subdirectory(albanyMesh)
add_subdirectory(HandsOn)
add_subdirectory(inputFiles)
add_subdirectory(mFiles)
add_subdirectory(ncGridSamples)

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyTPath} ${CMAKE_CURRENT_BINARY_DIR}/AlbanyT)

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${CismAlbanyPath} ${CMAKE_CURRENT_BINARY_DIR}/cism_driver)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_test.sh
               ${CMAKE_CURRENT_BINARY_DIR}/run_test.sh COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/process_output_test.sh
               ${CMAKE_CURRENT_BINARY_DIR}/process_output_test.sh COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.py
               ${CMAKE_CURRENT_BINARY_DIR}/runtest.py COPYONLY)
