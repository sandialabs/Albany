##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-single.g
               ${CMAKE_CURRENT_BINARY_DIR}/cube-single.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/larger-cubes.g.8.0
               ${CMAKE_CURRENT_BINARY_DIR}/larger-cubes.g.8.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/larger-cubes.g.8.1
               ${CMAKE_CURRENT_BINARY_DIR}/larger-cubes.g.8.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/larger-cubes.g.8.2
               ${CMAKE_CURRENT_BINARY_DIR}/larger-cubes.g.8.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/larger-cubes.g.8.3
               ${CMAKE_CURRENT_BINARY_DIR}/larger-cubes.g.8.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/larger-cubes.g.8.4
               ${CMAKE_CURRENT_BINARY_DIR}/larger-cubes.g.8.4 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/larger-cubes.g.8.5
               ${CMAKE_CURRENT_BINARY_DIR}/larger-cubes.g.8.5 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/larger-cubes.g.8.6
               ${CMAKE_CURRENT_BINARY_DIR}/larger-cubes.g.8.6 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/larger-cubes.g.8.7
               ${CMAKE_CURRENT_BINARY_DIR}/larger-cubes.g.8.7 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials-cubes.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials-cubes.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-single-piro.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cube-single-piro.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-single-piro-expl.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cube-single-piro-expl.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-single-tempus.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cube-single-tempus.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-single-tempus-expl.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cube-single-tempus-expl.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/larger-cubes-piro.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/larger-cubes-piro.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/larger-cubes-piro-expl.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/larger-cubes-piro-expl.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/larger-cubes-tempus.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/larger-cubes-tempus.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/larger-cubes-tempus-expl.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/larger-cubes-tempus-expl.yaml COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if (ALBANY_TEMPUS)
if (ALBANY_IFPACK2) 
add_test(Serial_Dynamic_${testName}_NewmarkBeta_Tempus ${SerialAlbanyT.exe} cube-single-tempus.yaml)
add_test(Parallel_Dynamic_${testName}_NewmarkBeta_Tempus ${AlbanyT8.exe} larger-cubes-tempus.yaml)
endif()
endif()

if (ALBANY_EPETRA) 
add_test(Serial_Dynamic_${testName}_NewmarkBeta_Piro ${SerialAlbany.exe} cube-single-piro.yaml)
add_test(Parallel_Dynamic_${testName}_NewmarkBeta_Piro ${Albany8.exe} larger-cubes-piro.yaml)
endif() 

