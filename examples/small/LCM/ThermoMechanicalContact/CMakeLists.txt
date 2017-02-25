
# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/elastic.xml
               ${CMAKE_CURRENT_BINARY_DIR}/elastic.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cyl.g
               ${CMAKE_CURRENT_BINARY_DIR}/cyl.g COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable

if (ALBANY_CONTACT)
   if (ALBANY_EPETRA)
   add_test(${testName} ${Albany.exe} input.xml)
   endif()

   # Note: This test will not run with the Tpetra executable as the trapezoidal
   # time integrator needs to be converted to tpetra.

   if (ALBANY_IFPACK2)
   #add_test(${testName}_Tpetra ${AlbanyT.exe} inputT.xml)
   endif ()
endif()
