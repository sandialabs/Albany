##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# This test is not currently set up to run on the BGQ...

# Create a symlink to the Partition executable
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${DTK_Interp_and_Error.exe} ${CMAKE_CURRENT_BINARY_DIR}/DTK_Interp_and_Error)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_disp.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_disp.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_cauchy_stress.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_cauchy_stress.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.py
               ${CMAKE_CURRENT_BINARY_DIR}/runtest.py COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
add_test(NAME ${testName} COMMAND "python" "runtest.py")

