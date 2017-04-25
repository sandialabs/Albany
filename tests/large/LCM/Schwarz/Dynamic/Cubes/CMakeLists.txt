##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-single.g
               ${CMAKE_CURRENT_BINARY_DIR}/cube-single.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials-single.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials-single.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-single-piro.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cube-single-piro.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-single-tempus.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cube-single-tempus.yaml COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if (ALBANY_TEMPUS)
if (ALBANY_IFPACK2) 
add_test(Dynamic_${testName}_NewmarkBeta_Tempus ${SerialAlbanyT.exe} cube-single-tempus.yaml)
endif()
endif()

if (ALBANY_EPETRA) 
add_test(Dynamic_${testName}_NewmarkBeta_Piro ${SerialAlbany.exe} cube-single-piro.yaml)
endif() 

