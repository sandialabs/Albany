#*****************************************************************//
#    Albany 2.0:  Copyright 2012 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//


# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTM.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputTM.xml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
IF(ENABLE_MESHMODEL)
  add_test(
      NAME ${testName}_TM_Mmodel
      COMMAND ${Albany.exe} inputTM.xml)
ENDIF()
