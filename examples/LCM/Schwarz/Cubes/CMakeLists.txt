##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.e
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1.e
               ${CMAKE_CURRENT_BINARY_DIR}/cube1.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-single.e
               ${CMAKE_CURRENT_BINARY_DIR}/cube-single.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials0.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials0.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials1.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials1.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials-single.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials-single.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.xml
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1.xml
               ${CMAKE_CURRENT_BINARY_DIR}/cube1.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cubes.xml
               ${CMAKE_CURRENT_BINARY_DIR}/cubes.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-single.xml
               ${CMAKE_CURRENT_BINARY_DIR}/cube-single.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtestT.py
               ${CMAKE_CURRENT_BINARY_DIR}/runtestT.py COPYONLY)

#create symlink to AlbanyT
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyTPath} ${CMAKE_CURRENT_BINARY_DIR}/AlbanyT)
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
# Teko needs Epetra
add_test(NAME Schwarz_${testName} COMMAND "python" "runtestT.py")
