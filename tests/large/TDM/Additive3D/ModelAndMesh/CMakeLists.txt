
# The root name for the tests in this directory is the directory name
get_filename_component(testNameRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Bare_plate0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/Bare_plate0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Bare_plate_4P0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/Bare_plate_4P0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Bare_plate_4P1.smb
               ${CMAKE_CURRENT_BINARY_DIR}/Bare_plate_4P1.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Bare_plate_4P2.smb
               ${CMAKE_CURRENT_BINARY_DIR}/Bare_plate_4P2.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Bare_plate_4P3.smb
               ${CMAKE_CURRENT_BINARY_DIR}/Bare_plate_4P3.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Bare_plate.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/Bare_plate.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Single_layer.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/Single_layer.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Single_layer_mesh0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/Single_layer_mesh0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Single_layer_mesh_1P0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/Single_layer_mesh_1P0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Single_layer_mesh_4P0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/Single_layer_mesh_4P0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Single_layer_mesh_4P1.smb
               ${CMAKE_CURRENT_BINARY_DIR}/Single_layer_mesh_4P1.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Single_layer_mesh_4P2.smb
               ${CMAKE_CURRENT_BINARY_DIR}/Single_layer_mesh_4P2.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Single_layer_mesh_4P3.smb
               ${CMAKE_CURRENT_BINARY_DIR}/Single_layer_mesh_4P3.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Single_layer_mesh.sms
               ${CMAKE_CURRENT_BINARY_DIR}/Single_layer_mesh.sms COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Single_layer_nat.x_t
               ${CMAKE_CURRENT_BINARY_DIR}/Single_layer_nat.x_t COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Single_layer.smd
               ${CMAKE_CURRENT_BINARY_DIR}/Single_layer.smd COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Single_layer.x_t
               ${CMAKE_CURRENT_BINARY_DIR}/Single_layer.x_t COPYONLY)

