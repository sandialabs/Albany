configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gauge.g
               ${CMAKE_CURRENT_BINARY_DIR}/gauge.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lower_bar.g
               ${CMAKE_CURRENT_BINARY_DIR}/lower_bar.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/upper_bar.g
               ${CMAKE_CURRENT_BINARY_DIR}/upper_bar.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gauge.xml
               ${CMAKE_CURRENT_BINARY_DIR}/gauge.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lower_bar.xml
               ${CMAKE_CURRENT_BINARY_DIR}/lower_bar.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/upper_bar.xml
               ${CMAKE_CURRENT_BINARY_DIR}/upper_bar.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gauge_material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/gauge_material.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lower_bar_material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/lower_bar_material.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/upper_bar_material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/upper_bar_material.xml COPYONLY)

add_subdirectory(matrixFree)
