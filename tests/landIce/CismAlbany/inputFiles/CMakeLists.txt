# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cism-albanyT.config
               ${CMAKE_CURRENT_BINARY_DIR}/cism-albanyT.config COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/create2dExo.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/create2dExo.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_cism-albanyT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_cism-albanyT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_standalone-albanyT_withFlowRate.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_standalone-albanyT_withFlowRate.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_standalone-albanyT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_standalone-albanyT.yaml COPYONLY)

