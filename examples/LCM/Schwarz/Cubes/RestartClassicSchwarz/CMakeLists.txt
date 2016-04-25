##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0_in.exo
               ${CMAKE_CURRENT_BINARY_DIR}/cube0_in.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1_in.exo
               ${CMAKE_CURRENT_BINARY_DIR}/cube1_in.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0_restart.xml
               ${CMAKE_CURRENT_BINARY_DIR}/cube0_restart.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1_restart.xml
               ${CMAKE_CURRENT_BINARY_DIR}/cube1_restart.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/classical_schwarz.sh
               ${CMAKE_CURRENT_BINARY_DIR}/classical_schwarz.sh COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_schwarz_cube1_target.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_schwarz_cube1_target.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_schwarz_cube0_target.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_schwarz_cube0_target.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/norm_displacements.m
               ${CMAKE_CURRENT_BINARY_DIR}/norm_displacements.m COPYONLY)

#create symlink to AlbanyT
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyTPath} ${CMAKE_CURRENT_BINARY_DIR}/AlbanyT)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${DTK_Interp_Volume_to_NS.exe} ${CMAKE_CURRENT_BINARY_DIR}/DTK_Interp_Volume_to_NS)

#get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
# Teko needs Epetra
#add_test(NAME Schwarz_${testName} COMMAND "python" "runtestT.py")
