##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cube.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-loca-time-dept-bc-sdbcs.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cube-loca-time-dept-bc-sdbcs.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube-loca-time-dept-bc-dbcs.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/cube-loca-time-dept-bc-dbcs.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.py
               ${CMAKE_CURRENT_BINARY_DIR}/runtest.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.g
               ${CMAKE_CURRENT_BINARY_DIR}/cube.g COPYONLY)

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyTPath} ${CMAKE_CURRENT_BINARY_DIR}/AlbanyT)
if (ALBANY_IFPACK2) 
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
add_test(NAME ${testName} COMMAND "python" "runtest.py")
add_test(${testName}_TimeDeptBC ${SerialAlbanyT.exe} cube-loca-time-dept-bc-sdbcs.yaml)
endif()
