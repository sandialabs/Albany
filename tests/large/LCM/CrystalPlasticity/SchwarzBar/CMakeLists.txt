configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gauge.g
               ${CMAKE_CURRENT_BINARY_DIR}/gauge.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lower_bar.g
               ${CMAKE_CURRENT_BINARY_DIR}/lower_bar.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/upper_bar.g
               ${CMAKE_CURRENT_BINARY_DIR}/upper_bar.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gauge.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/gauge.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lower_bar.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/lower_bar.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/upper_bar.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/upper_bar.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gauge_material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/gauge_material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lower_bar_material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/lower_bar_material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/upper_bar_material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/upper_bar_material.yaml COPYONLY)

add_subdirectory(matrixFree)
add_subdirectory(modified)
