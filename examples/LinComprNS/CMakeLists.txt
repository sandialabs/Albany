

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input2D.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input3D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input3D.xml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
add_test(${testName}_2D ${Albany.exe} input2D.xml)
add_test(${testName}_3D ${Albany.exe} input3D.xml)

