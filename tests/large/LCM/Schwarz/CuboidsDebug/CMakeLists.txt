##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

add_subdirectory(SymmetryBC)
add_subdirectory(NoSymmetryBC)
# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CoupledSchwarz_Cubes_StudyOverlap_Cube0.g
               ${CMAKE_CURRENT_BINARY_DIR}/CoupledSchwarz_Cubes_StudyOverlap_Cube0.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CoupledSchwarz_Cubes_StudyOverlap_Cube1.g
               ${CMAKE_CURRENT_BINARY_DIR}/CoupledSchwarz_Cubes_StudyOverlap_Cube1.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CoupledSchwarz_Cubes_StudyOverlap_Cube0_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/CoupledSchwarz_Cubes_StudyOverlap_Cube0_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CoupledSchwarz_Cubes_StudyOverlap_Cube1_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/CoupledSchwarz_Cubes_StudyOverlap_Cube1_Material.yaml COPYONLY)

#create symlink to AlbanyT
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyTPath} ${CMAKE_CURRENT_BINARY_DIR}/AlbanyT)
