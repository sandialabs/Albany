##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.e
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1.e
               ${CMAKE_CURRENT_BINARY_DIR}/cube1.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-single.e
               ${CMAKE_CURRENT_BINARY_DIR}/cube-single.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials0.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials0.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials1.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials1.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials-single.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials-single.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cube1.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cubes.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cubes.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-single.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cube-single.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/check_comparisons.py
               ${CMAKE_CURRENT_BINARY_DIR}/check_comparisons.py COPYONLY)

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${runtest.cmake} ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyTPath} ${CMAKE_CURRENT_BINARY_DIR}/AlbanyT)

#add_test(NAME Schwarz_${testName} COMMAND "python" "runtestT.py")

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
#test 1 - Cubes DBC 
SET(OUTFILE "Cubes_DBC.log")
SET(PYTHON_FILE "check_comparisons.py") 
add_test(NAME Schwarz_${testName}
        COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
        -DTEST_NAME=Cubes -DTEST_ARGS=cubes.yaml -DMPIMNP=1
        -DLOGFILE=${OUTFILE} -DPY_FILE=${PYTHON_FILE}
        -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})

