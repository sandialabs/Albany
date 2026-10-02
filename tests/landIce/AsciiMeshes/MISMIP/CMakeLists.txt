configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mismip_2d.exo
               ${CMAKE_CURRENT_BINARY_DIR}/mismip_2d.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface_height.ascii
               ${CMAKE_CURRENT_BINARY_DIR}/surface_height.ascii COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/averaged_velocity.ascii
               ${CMAKE_CURRENT_BINARY_DIR}/averaged_velocity.ascii COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thickness.ascii
               ${CMAKE_CURRENT_BINARY_DIR}/thickness.ascii COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/bed_topography.ascii
               ${CMAKE_CURRENT_BINARY_DIR}/bed_topography.ascii COPYONLY)