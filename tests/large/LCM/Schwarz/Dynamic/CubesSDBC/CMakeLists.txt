##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.g
               ${CMAKE_CURRENT_BINARY_DIR}/cube.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cube.yaml COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if (ALBANY_TEMPUS)
if (ALBANY_IFPACK2) 
add_test(Serial_Dynamic_${testName}_NewmarkImplicitDForm_Tempus ${SerialAlbanyT.exe} cube.yaml)
endif()
endif()
