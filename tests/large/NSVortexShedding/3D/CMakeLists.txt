# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTransientRK.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputTransientRK.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vortex3D.exo
               ${CMAKE_CURRENT_BINARY_DIR}/vortex3D.exo COPYONLY)
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/data.perf
#               ${CMAKE_CURRENT_BINARY_DIR}/data.perf COPYONLY)


# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
#add_test(${testName}_perf ${performanceTestScript})
