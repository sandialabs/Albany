##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# This test is not currently set up to run on the BGQ...

# Create a symlink to the Partition executable
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${DTK_Interp_and_Error.exe} ${CMAKE_CURRENT_BINARY_DIR}/DTK_Interp_and_Error)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${runtest.cmake} ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_disp.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_disp.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_cauchy_stress.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_cauchy_stress.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/check_comparisons.py
               ${CMAKE_CURRENT_BINARY_DIR}/check_comparisons.py COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
SET(OUTFILE "DTKInterpNotchedCyl.log")
SET(PYTHON_FILE "check_comparisons.py") 
add_test(NAME DTKInterp_${testName}
        COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Parallel_DTK_Interp_and_Error.exe}"
        -DTEST_NAME=Error -DTEST_ARGS=--yaml-in-file=input_disp.yaml -DMPIMNP=${MPIMNP}
        -DLOGFILE=${OUTFILE} -DPY_FILE=${PYTHON_FILE}
        -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})

