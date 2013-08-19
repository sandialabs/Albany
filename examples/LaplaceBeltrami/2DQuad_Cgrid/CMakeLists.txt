
# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTPS.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputTPS.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputLB.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputLB.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2d_cgrid_quad4.e
               ${CMAKE_CURRENT_BINARY_DIR}/2d_cgrid_quad4.e COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
add_test(LaplaceBeltrami:${testName} ${Albany.exe})
add_test(LaplaceBeltrami:${testName}_TPS ${Albany.exe} inputTPS.xml)
add_test(LaplaceBeltrami:${testName}_LB ${Albany.exe} inputLB.xml)
