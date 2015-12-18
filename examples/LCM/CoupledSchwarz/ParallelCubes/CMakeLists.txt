##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
if (ALBANY_DTK)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.exo
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.exo.2.0
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.exo.2.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.exo.2.1
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.exo.2.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1_ns1.exo
               ${CMAKE_CURRENT_BINARY_DIR}/cube1_ns1.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1_ns1.exo.2.0
               ${CMAKE_CURRENT_BINARY_DIR}/cube1_ns1.exo.2.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1_ns1.exo.2.1
               ${CMAKE_CURRENT_BINARY_DIR}/cube1_ns1.exo.2.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0-dtk.xml
               ${CMAKE_CURRENT_BINARY_DIR}/cube0-dtk.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1-dtk.xml
               ${CMAKE_CURRENT_BINARY_DIR}/cube1-dtk.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cubes-dtk.xml
               ${CMAKE_CURRENT_BINARY_DIR}/cubes-dtk.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials0.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials0.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials1.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials1.xml COPYONLY)
endif()

