##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.e
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.e.8.0
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.e.8.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.e.8.1
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.e.8.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.e.8.2
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.e.8.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.e.8.3
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.e.8.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.e.8.4
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.e.8.4 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.e.8.5
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.e.8.5 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.e.8.6
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.e.8.6 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.e.8.7
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.e.8.7 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1.e
               ${CMAKE_CURRENT_BINARY_DIR}/cube1.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1.e.8.0
               ${CMAKE_CURRENT_BINARY_DIR}/cube1.e.8.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1.e.8.1
               ${CMAKE_CURRENT_BINARY_DIR}/cube1.e.8.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1.e.8.2
               ${CMAKE_CURRENT_BINARY_DIR}/cube1.e.8.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1.e.8.3
               ${CMAKE_CURRENT_BINARY_DIR}/cube1.e.8.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1.e.8.4
               ${CMAKE_CURRENT_BINARY_DIR}/cube1.e.8.4 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1.e.8.5
               ${CMAKE_CURRENT_BINARY_DIR}/cube1.e.8.5 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1.e.8.6
               ${CMAKE_CURRENT_BINARY_DIR}/cube1.e.8.6 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1.e.8.7
               ${CMAKE_CURRENT_BINARY_DIR}/cube1.e.8.7 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials0.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials0.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials1.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials1.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.xml
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1.xml
               ${CMAKE_CURRENT_BINARY_DIR}/cube1.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cubes.xml
               ${CMAKE_CURRENT_BINARY_DIR}/cubes.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtestT.py
               ${CMAKE_CURRENT_BINARY_DIR}/runtestT.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtestT_dtk_8proc.py
               ${CMAKE_CURRENT_BINARY_DIR}/runtestT_dtk_8proc.py COPYONLY)

#create symlink to AlbanyT
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyTPath} ${CMAKE_CURRENT_BINARY_DIR}/AlbanyT)

if(ALBANY_DTK)
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
add_test(NAME CoupledSchwarz_${testName}_DTK_8proc COMMAND "python" "runtestT_dtk_8proc.py")
endif()
