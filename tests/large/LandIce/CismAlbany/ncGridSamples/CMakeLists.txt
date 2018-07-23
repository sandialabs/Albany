# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/greenland.nc
               ${CMAKE_CURRENT_BINARY_DIR}/greenland.nc COPYONLY)

