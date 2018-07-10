##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.e
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.e.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.e.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.e.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.e.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.e.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.e.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.e.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.e.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1.e
               ${CMAKE_CURRENT_BINARY_DIR}/cube1.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1.e.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/cube1.e.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1.e.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/cube1.e.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1.e.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/cube1.e.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1.e.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/cube1.e.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials0.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials0.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials1.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials1.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cube0.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube0_sdbc.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cube0_sdbc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cube1.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube1_sdbc.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cube1_sdbc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cubes.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cubes.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cubes_sdbc.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cubes_sdbc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/check_comparisons_dbc.py
               ${CMAKE_CURRENT_BINARY_DIR}/check_comparisons_dbc.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/check_comparisons_sdbc.py
               ${CMAKE_CURRENT_BINARY_DIR}/check_comparisons_sdbc.py COPYONLY)

#create symlink to AlbanyT
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyTPath} ${CMAKE_CURRENT_BINARY_DIR}/AlbanyT)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${runtest.cmake} ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake)


if(ALBANY_DTK)
# 3. Create the test with this name and standard executable
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
SET(OUTFILE "Parallel_Cubes_4_DBC.Log")
SET(PYTHON_FILE "check_comparisons_dbc.py")
add_test(NAME Schwarz_${testName}_DTK_DBC_np4
        COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
        -DTEST_NAME=Cubes -DTEST_ARGS=cubes.yaml -DMPIMNP=4
        -DLOGFILE=${OUTFILE} -DPY_FILE=${PYTHON_FILE}
        -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})
SET(OUTFILE "Parallel_Cubes_4_SDBC.Log")
SET(PYTHON_FILE "check_comparisons_sdbc.py")
add_test(NAME Schwarz_${testName}_DTK_SDBC_np4
        COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
        -DTEST_NAME=CubesSDBC -DTEST_ARGS=cubes_sdbc.yaml -DMPIMNP=4
        -DLOGFILE=${OUTFILE} -DPY_FILE=${PYTHON_FILE}
        -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})
endif()
