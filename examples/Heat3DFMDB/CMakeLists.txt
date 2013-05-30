#*****************************************************************//
#    Albany 2.0:  Copyright 2012 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//


# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSerial.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSerial.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMmodel.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMmodel.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_non_man.xmt_txt
               ${CMAKE_CURRENT_BINARY_DIR}/test_non_man.xmt_txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coarse_cube_mesh_fmdb0.sms
               ${CMAKE_CURRENT_BINARY_DIR}/coarse_cube_mesh_fmdb0.sms COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coarse_cube_mesh_fmdb1.sms
               ${CMAKE_CURRENT_BINARY_DIR}/coarse_cube_mesh_fmdb1.sms COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coarse_cube_mesh_fmdb2.sms
               ${CMAKE_CURRENT_BINARY_DIR}/coarse_cube_mesh_fmdb2.sms COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coarse_cube_mesh_fmdb3.sms
               ${CMAKE_CURRENT_BINARY_DIR}/coarse_cube_mesh_fmdb3.sms COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/serial_coarse_cube_mesh_fmdb0.sms
               ${CMAKE_CURRENT_BINARY_DIR}/serial_coarse_cube_mesh_fmdb0.sms COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable

add_test(${testName} ${Albany.exe})

IF(NOT ALBANY_PARALLEL_ONLY)
  add_test(${testName}_SERIAL ${SerialAlbany.exe} inputSerial.xml)
ENDIF()

add_test(${testName}_Mmodel ${Albany.exe} inputMmodel.xml)

