# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cism-albanyT.config
               ${CMAKE_CURRENT_BINARY_DIR}/cism-albanyT.config COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/create2dExo.xml
               ${CMAKE_CURRENT_BINARY_DIR}/create2dExo.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_cism-albanyT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_cism-albanyT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_standalone-albanyT_withFlowRate.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_standalone-albanyT_withFlowRate.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_standalone-albanyT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_standalone-albanyT.xml COPYONLY)

