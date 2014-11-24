

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_test2D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_test2D.xml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_test2D NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName} ${Albany.exe} input_test2D.xml)
endif()

if (ALBANY_IFPACK2)
# 1'. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_test2DT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_test2DT.xml COPYONLY)
# 2'. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_test2D NAME)
# 3'. Create the test with this name and standard executable
add_test(${testName}_Tpetra ${AlbanyT.exe} input_test2DT.xml)
endif ()
