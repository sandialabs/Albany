#*****************************************************************//
#    Albany 2.0:  Copyright 2012 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputP1.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputP1.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputP2.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputP2.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAdjoint.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputAdjoint.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAdjointEnriched.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputAdjointEnriched.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)

# 2. Name the test with the directory name, prepended by GOAL_
get_filename_component(testRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(testName "GOAL_${testRoot}")

# 3. Create the test with this name and standard executable
IF(ENABLE_MESHMODEL)
  IF(ALBANY_IFPACK2)
    add_test(NAME ${testName}_P1 COMMAND ${AlbanyT.exe} inputP1.xml)
    add_test(NAME ${testName}_P2 COMMAND ${AlbanyT.exe} inputP2.xml)
    add_test(NAME ${testName}_Adjoint COMMAND ${AlbanyT.exe} inputAdjoint.xml)
    add_test(NAME ${testName}_AdjointEnriched COMMAND ${AlbanyT.exe} inputAdjointEnriched.xml)
  ENDIF()
ENDIF()
