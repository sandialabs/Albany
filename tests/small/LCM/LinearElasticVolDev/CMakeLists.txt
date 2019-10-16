##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.g
  ${CMAKE_CURRENT_BINARY_DIR}/cube.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/cube.yaml COPYONLY)

set(runtest.cmake ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${runtest.cmake} ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyPath} ${CMAKE_CURRENT_BINARY_DIR}/Albany)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
#test 1
SET(OUTFILE "cube.log")
SET(PYTHON_FILE "check_comparisons.py") 
add_test(NAME ${testName}
        COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}"
        -DTEST_NAME=Cubes -DTEST_ARGS=cube.yaml -DMPIMNP=1
        -DLOGFILE=${OUTFILE} -DPY_FILE=${PYTHON_FILE}
        -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})
set_tests_properties(${testName} PROPERTIES LABELS "LCM;Tpetra;Forward")

