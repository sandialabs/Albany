##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# This test is not currently set up to run on the BGQ...

# Create a symlink to the Partition executable

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${DTK_Interp_Volume_to_NS.exe} ${CMAKE_CURRENT_BINARY_DIR}/DTK_Interp_Volume_to_NS)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${runtest.cmake} ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_schwarz.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_schwarz.yaml COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
add_test(DTKInterp_${testName} ${Parallel_DTK_Interp_Volume_to_NS.exe} --yaml-in-file=input_schwarz.yaml)

