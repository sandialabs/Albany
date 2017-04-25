
# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/elastic.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/elastic.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cyl.g
               ${CMAKE_CURRENT_BINARY_DIR}/cyl.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2dsmall.e
               ${CMAKE_CURRENT_BINARY_DIR}/2dsmall.e COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable

if (ALBANY_CONTACT)

   # Note: This test will not run with the Tpetra executable as the trapezoidal
   # time integrator needs to be converted to tpetra.

   if (ALBANY_IFPACK2)
   #add_test(${testName} ${AlbanyT.exe} input.yaml)
   endif ()
endif()
