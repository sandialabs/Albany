##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coarse0.g
               ${CMAKE_CURRENT_BINARY_DIR}/coarse0.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coarse1.g
               ${CMAKE_CURRENT_BINARY_DIR}/coarse1.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coarse0.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/coarse0.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coarse1.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/coarse1.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coarses.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/coarses.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials0.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials0.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials1.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials1.yaml COPYONLY)

