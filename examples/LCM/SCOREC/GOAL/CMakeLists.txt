#*****************************************************************//
#    Albany 2.0:  Copyright 2012 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputElasticT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputElasticT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/elastic.xml
               ${CMAKE_CURRENT_BINARY_DIR}/elastic.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputJ2T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputJ2T.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/j2.xml
               ${CMAKE_CURRENT_BINARY_DIR}/j2.xml COPYONLY)

# 2. Name the test with the directory name, prepended by SCOREC_
get_filename_component(testRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(testName "SCOREC_${testRoot}")

# 3. Create the test with this name and standard executable
IF(ENABLE_MESHMODEL)
  IF(ALBANY_IFPACK2)
    add_test(NAME ${testName}_Elastic_Tpetra COMMAND ${AlbanyT.exe} inputElasticT.xml)
    add_test(NAME ${testName}_J2_Tpetra COMMAND ${AlbanyT.exe} inputJ2T.xml)
  ENDIF()
ENDIF()
