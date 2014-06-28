#*****************************************************************//
#    Albany 2.0:  Copyright 2012 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//


# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube_pumi.sms
               ${CMAKE_CURRENT_BINARY_DIR}/cube_pumi.sms COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.sat
               ${CMAKE_CURRENT_BINARY_DIR}/cube.sat COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable

IF(ENABLE_ACIS)
  add_test(NAME ${testName}
     COMMAND ${CMAKE_COMMAND}  "-DTEST_PROG=${Albany.exe}" -DTEST_ARGS= )
ENDIF()
