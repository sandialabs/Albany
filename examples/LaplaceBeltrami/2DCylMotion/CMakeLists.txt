
# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputDTPS.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputDTPS.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTPSA.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputTPSA.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputDLB.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputDLB.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cyl.exo
               ${CMAKE_CURRENT_BINARY_DIR}/cyl.exo COPYONLY)

# 4. Name the test with the directory name
get_filename_component(testDirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 5. Create the test with this name and standard executable
set(testName ${testDirName})

add_test(LaplaceBeltrami:${testName}_DTPS ${Albany.exe} inputDTPS.xml)
