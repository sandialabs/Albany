add_subdirectory(albanyMesh)
add_subdirectory(HandsOn)
add_subdirectory(inputFiles)
add_subdirectory(mFiles)
add_subdirectory(ncGridSamples)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_test.sh
               ${CMAKE_CURRENT_BINARY_DIR}/run_test.sh COPYONLY)
