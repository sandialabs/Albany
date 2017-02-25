# 1. Copy Input file from source to binary dir

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
#  First test 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_dome.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_dome.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_Dome ${Albany.exe} input_dome.xml)
endif()

