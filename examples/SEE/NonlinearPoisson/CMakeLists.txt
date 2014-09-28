#*****************************************************************//
#    Albany 2.0:  Copyright 2012 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//

# 1. Copy input file from source dirs to binary dirs
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/cube.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cube1.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube2.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cube2.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube3.smb
               ${CMAKE_CURRENT_BINARY_DIR}/cube3.smb COPYONLY)

# 2. Name the test with the directory name, prepended by SEE_
get_filename_component(test_root ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(test_name "SEE_${test_root}")

# 3. Create the test with this name and standard executable
add_test(${test_name}_Tpetra ${AlbanyT.exe} input.xml)
