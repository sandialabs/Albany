##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputCoupledSchwarzT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputCoupledSchwarzT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputCoupledSchwarz_Teko_BGS_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputCoupledSchwarz_Teko_BGS_T.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputLOCAT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputLOCAT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputLOCAT0.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputLOCAT0.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputLOCAT1.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputLOCAT1.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputCoupledSchwarzLOCAT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputCoupledSchwarzLOCAT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT0.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT0.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT1.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT1.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials0.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials0.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials1.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials1.yaml COPYONLY)

add_subdirectory(Cubes)
add_subdirectory(CuboidsDebug)
if(ALBANY_DTK)
add_subdirectory(ParallelCubes)
endif()
add_subdirectory(NotchedCylinder)
add_subdirectory(LaserWeld)

