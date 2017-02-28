##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materialLaserWeldGauge.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materialLaserWeldGauge.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materialLaserWeldHolder0.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materialLaserWeldHolder0.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materialLaserWeldHolder1.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materialLaserWeldHolder1.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/modelLaserWeldGauge.xml
               ${CMAKE_CURRENT_BINARY_DIR}/modelLaserWeldGauge.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/modelLaserWeldHolder0.xml
               ${CMAKE_CURRENT_BINARY_DIR}/modelLaserWeldHolder0.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/modelLaserWeldHolder1.xml
               ${CMAKE_CURRENT_BINARY_DIR}/modelLaserWeldHolder1.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/specimenLaserWeldGauge.g
               ${CMAKE_CURRENT_BINARY_DIR}/specimenLaserWeldGauge.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/specimenLaserWeldHolder0.g
               ${CMAKE_CURRENT_BINARY_DIR}/specimenLaserWeldHolder0.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/specimenLaserWeldHolder1.g
               ${CMAKE_CURRENT_BINARY_DIR}/specimenLaserWeldHolder1.g COPYONLY)
