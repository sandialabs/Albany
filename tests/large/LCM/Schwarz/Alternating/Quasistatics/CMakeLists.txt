##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_00.g
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_00.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_00.g.2.0
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_00.g.2.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_00.g.2.1
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_00.g.2.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_01.g
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_01.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_01.g.2.0
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_01.g.2.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_01.g.2.1
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_01.g.2.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_00.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_00.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboid_01.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cuboid_01.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuboids.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cuboids.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_00.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials_00.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials_01.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials_01.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/check_convergence.py
               ${CMAKE_CURRENT_BINARY_DIR}/check_convergence.py COPYONLY)

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyTPath} ${CMAKE_CURRENT_BINARY_DIR}/AlbanyT)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${runtest.cmake} ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
SET(OUTFILE "cuboid.log")
SET(PYTHON_FILE "check_convergence.py") 
add_test(NAME Schwarz_Alternating_Dynamics_${testName}
        COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
        -DTEST_NAME=Cubes -DTEST_ARGS=cuboids.yaml -DMPIMNP=1
        -DLOGFILE=${OUTFILE} -DPY_FILE=${PYTHON_FILE}
        -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})

