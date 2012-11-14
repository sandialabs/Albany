

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTR.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputTR.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputLumped.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputLumped.xml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
add_test(${testName}_explicit ${Albany.exe})
add_test(${testName}_implicit ${Albany.exe} inputTR.xml)
add_test(${testName}_lumped   ${Albany.exe} inputLumped.xml)
