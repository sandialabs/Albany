##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# This test is not currently set up to run on the BGQ...

# Create a symlink to the Partition executable
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${DTK_Interp_Volume_to_NS.exe} ${CMAKE_CURRENT_BINARY_DIR}/DTK_Interp_Volume_to_NS)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_schwarz.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_schwarz.xml COPYONLY)

