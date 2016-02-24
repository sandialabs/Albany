##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coarse0.exo
               ${CMAKE_CURRENT_BINARY_DIR}/coarse0.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coarse1.exo
               ${CMAKE_CURRENT_BINARY_DIR}/coarse1.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coarse0.xml
               ${CMAKE_CURRENT_BINARY_DIR}/coarse0.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coarse1.xml
               ${CMAKE_CURRENT_BINARY_DIR}/coarse1.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coarses.xml
               ${CMAKE_CURRENT_BINARY_DIR}/coarses.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials0.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials0.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials1.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials1.xml COPYONLY)

