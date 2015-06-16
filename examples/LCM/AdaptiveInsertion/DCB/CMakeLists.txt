##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dcb-hex.jou
               ${CMAKE_CURRENT_BINARY_DIR}/dcb-hex.jou COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dcb-hex.g
               ${CMAKE_CURRENT_BINARY_DIR}/dcb-hex.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dcb-hex.xml
               ${CMAKE_CURRENT_BINARY_DIR}/dcb-hex.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mat-dcb.xml
               ${CMAKE_CURRENT_BINARY_DIR}/mat-dcb.xml COPYONLY)

