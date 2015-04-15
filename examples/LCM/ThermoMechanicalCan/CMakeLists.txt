##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Mechanics
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Mechanics.xml
               ${CMAKE_CURRENT_BINARY_DIR}/Mechanics.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsMaterials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsMaterials.xml COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
add_test(${testName} ${Albany.exe} Mechanics.xml)
set_tests_properties(${testName}  PROPERTIES LABELS CUDA_TEST)
add_test(${testName} ${AlbanyT.exe} MechanicsT.xml)
