
# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/elastic.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/elastic.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cyl.g
               ${CMAKE_CURRENT_BINARY_DIR}/cyl.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2dsmall.e
               ${CMAKE_CURRENT_BINARY_DIR}/2dsmall.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2dsmall.e.2.0
               ${CMAKE_CURRENT_BINARY_DIR}/2dsmall.e.2.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2dsmall.e.2.1
               ${CMAKE_CURRENT_BINARY_DIR}/2dsmall.e.2.1 COPYONLY)

# Create symlink to Albany for convenience
if (ALBANY_IFPACK2)
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${AlbanyPath} ${CMAKE_CURRENT_BINARY_DIR}/Albany)
endif()

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable

# Turn on test when it is ready

#if (ALBANY_CONTACT)
#
#   if (ALBANY_IFPACK2)
#     add_test(${testName} ${Albany.exe} input.yaml)
#   endif ()
#
#endif()
