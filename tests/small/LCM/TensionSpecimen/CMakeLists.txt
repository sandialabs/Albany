# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tension_hexes.g
               ${CMAKE_CURRENT_BINARY_DIR}/tension_hexes.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tension_dynamic_impl.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/tension_dynamic_impl.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tension_quasistatic.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/tension_quasistatic.yaml COPYONLY)
