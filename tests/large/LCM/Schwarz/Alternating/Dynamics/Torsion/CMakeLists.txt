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
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest_parallel.py
               ${CMAKE_CURRENT_BINARY_DIR}/runtest_parallel.py COPYONLY)
ENDIF()

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyTPath} ${CMAKE_CURRENT_BINARY_DIR}/AlbanyT)
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
IF(ALBANY_DTK)
add_test(NAME Schwarz_Alternating_Dynamics_${testName}_Parallel COMMAND "python" "runtest_parallel.py")
ENDIF()
