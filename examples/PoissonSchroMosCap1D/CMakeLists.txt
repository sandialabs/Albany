

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSchroPo.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSchroPo.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/poissonMOSCap.xml
               ${CMAKE_CURRENT_BINARY_DIR}/poissonMOSCap.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/schrodMOSCap.xml
               ${CMAKE_CURRENT_BINARY_DIR}/schrodMOSCap.xml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and executable
add_test(${testName} ${Albany.exe} inputSchroPo.xml)
