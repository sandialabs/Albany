##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

if (ALBANY_IFPACK2)
# Create a symlink to AlbanyT
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyTPath} ${CMAKE_CURRENT_BINARY_DIR}/AlbanyT)

# input files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.xml
  ${CMAKE_CURRENT_BINARY_DIR}/inputT.xml COPYONLY)

# material files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SimpleMaterials.xml
  ${CMAKE_CURRENT_BINARY_DIR}/SimpleMaterials.xml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
add_test(${testName} ${AlbanyT.exe} inputT.xml)
endif()




