
# The root name for the tests in this directory is the directory name
get_filename_component(testNameRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)


# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/40x20x5um_nat.x_t
               ${CMAKE_CURRENT_BINARY_DIR}/40x20x5um_nat.x_t COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/40x20x5umSi.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/40x20x5umSi.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/40x20x5umSi_nat.x_t 
               ${CMAKE_CURRENT_BINARY_DIR}/40x20x5umSi_nat.x_t COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/40x20x5umSi.smd 
               ${CMAKE_CURRENT_BINARY_DIR}/40x20x5umSi.smd COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/40x20x5um.x_t 
               ${CMAKE_CURRENT_BINARY_DIR}/40x20x5um.x_t COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Depth0.smb 
               ${CMAKE_CURRENT_BINARY_DIR}/Depth0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Depth_4P0.smb 
               ${CMAKE_CURRENT_BINARY_DIR}/Depth_4P0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Depth_4P1.smb 
               ${CMAKE_CURRENT_BINARY_DIR}/Depth_4P1.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Depth_4P2.smb 
               ${CMAKE_CURRENT_BINARY_DIR}/Depth_4P2.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Depth_4P3.smb 
               ${CMAKE_CURRENT_BINARY_DIR}/Depth_4P3.smb COPYONLY)
