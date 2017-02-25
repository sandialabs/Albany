# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/README
               ${CMAKE_CURRENT_BINARY_DIR}/README COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CHECKLIST
               ${CMAKE_CURRENT_BINARY_DIR}/CHECKLIST COPYONLY)

