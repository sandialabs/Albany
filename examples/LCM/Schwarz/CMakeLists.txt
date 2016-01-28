##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputCoupledSchwarzT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputCoupledSchwarzT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputCoupledSchwarz_Teko_BGS_T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputCoupledSchwarz_Teko_BGS_T.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputLOCAT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputLOCAT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputLOCAT0.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputLOCAT0.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputLOCAT1.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputLOCAT1.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputCoupledSchwarzLOCAT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputCoupledSchwarzLOCAT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT0.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT0.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT1.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT1.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials0.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials0.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials1.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials1.xml COPYONLY)

add_subdirectory(Cubes)
if(ALBANY_DTK)
add_subdirectory(ParallelCubes)
endif()
add_subdirectory(NotchedCylinder)
add_subdirectory(LaserWeld)

