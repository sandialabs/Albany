##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dynamics.xml
               ${CMAKE_CURRENT_BINARY_DIR}/dynamics.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/elastic.xml
               ${CMAKE_CURRENT_BINARY_DIR}/elastic.xml COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
add_test(${testName} ${Albany.exe} dynamics.xml)