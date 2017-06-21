##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CoupledSchwarz_Cubes_StudyOverlap_Cube0_Model.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/CoupledSchwarz_Cubes_StudyOverlap_Cube0_Model.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CoupledSchwarz_Cubes_StudyOverlap_Cube1_Model.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/CoupledSchwarz_Cubes_StudyOverlap_Cube1_Model.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CoupledSchwarz_Cubes_StudyOverlap.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/CoupledSchwarz_Cubes_StudyOverlap.yaml COPYONLY)
