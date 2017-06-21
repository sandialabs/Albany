##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materialLaserWeldGauge.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materialLaserWeldGauge.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materialLaserWeldHolder0.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materialLaserWeldHolder0.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materialLaserWeldHolder1.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materialLaserWeldHolder1.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/modelLaserWeldGauge.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/modelLaserWeldGauge.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/modelLaserWeldHolder0.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/modelLaserWeldHolder0.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/modelLaserWeldHolder1.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/modelLaserWeldHolder1.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/specimenLaserWeldGauge.g
               ${CMAKE_CURRENT_BINARY_DIR}/specimenLaserWeldGauge.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/specimenLaserWeldHolder0.g
               ${CMAKE_CURRENT_BINARY_DIR}/specimenLaserWeldHolder0.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/specimenLaserWeldHolder1.g
               ${CMAKE_CURRENT_BINARY_DIR}/specimenLaserWeldHolder1.g COPYONLY)
