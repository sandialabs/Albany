##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid.g
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid.xml
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.py
               ${CMAKE_CURRENT_BINARY_DIR}/runtest.py COPYONLY)

# Create symlink to AlbanyT
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyTPath} ${CMAKE_CURRENT_BINARY_DIR}/AlbanyT)
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Create the test with this name and standard executable
add_test(NAME ${testName} COMMAND "python" "runtest.py")
