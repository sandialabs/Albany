##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nobcs_00.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/nobcs_00.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nobcs_01.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/nobcs_01.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nobcs.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/nobcs.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/material_00.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/material_00.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/material_01.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/material_01.yaml COPYONLY)
IF(ALBANY_DTK)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/check_convergence.py
               ${CMAKE_CURRENT_BINARY_DIR}/check_convergence.py COPYONLY)
ENDIF()

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyTPath} ${CMAKE_CURRENT_BINARY_DIR}/AlbanyT)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${runtest.cmake} ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
IF(ALBANY_DTK)
SET(OUTFILE "torsion_parallel.log")
SET(PYTHON_FILE "check_convergence.py") 
add_test(NAME Schwarz_Alternating_Dynamics_${testName}_Parallel
        COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
        -DTEST_NAME=TorsionParallel -DTEST_ARGS=nobcs.yaml -DMPIMNP=4
        -DLOGFILE=${OUTFILE} -DPY_FILE=${PYTHON_FILE}
        -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})
ENDIF()
