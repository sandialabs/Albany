#*****************************************************************//
#    Albany 2.0:  Copyright 2012 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//


# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMmodel.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMmodel.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/unit_cube.xmt_txt
               ${CMAKE_CURRENT_BINARY_DIR}/unit_cube.xmt_txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/unit_cube_mmodel.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/unit_cube_mmodel.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/unit_cube_fmdb.sms
               ${CMAKE_CURRENT_BINARY_DIR}/unit_cube_fmdb.sms COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable

IF(ENABLE_PARASOLID)
  add_test(${testName} ${Albany.exe})
ENDIF()

IF(ENABLE_MESHMODEL)
  add_test(${testName}_Mmodel ${Albany.exe} inputMmodel.xml)
ENDIF()
