##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-0.g
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-0.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-1.g
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-1.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-0.xml
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-0.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-1.xml
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-1.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-S.xml
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-S.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials-0.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials-0.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials-1.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials-1.xml COPYONLY)

