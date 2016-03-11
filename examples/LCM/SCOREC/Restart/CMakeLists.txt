#*****************************************************************//
#    Albany 2.0:  Copyright 2012 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputNoRestartT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputNoRestartT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputWriteRestartT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputWriteRestartT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputRestartFromFileT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputRestartFromFileT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/assoc.txt
               ${CMAKE_CURRENT_BINARY_DIR}/assoc.txt COPYONLY)

# 2. Name the test with the directory name, prepended by SCOREC_
get_filename_component(testRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(testName "SCOREC_${testRoot}")

# 3. Create the test with this name and standard executable
IF(ENABLE_MESHMODEL)
  IF(ALBANY_IFPACK2)
    add_test(NAME ${testName}_NoRestartT COMMAND ${AlbanyT.exe} inputNoRestartT.xml)
    add_test(NAME ${testName}_WriteRestartT COMMAND ${AlbanyT.exe} inputWriteRestartT.xml)
    add_test(NAME ${testName}_RestartFromFileT COMMAND ${AlbanyT.exe} inputRestartFromFileT.xml)
    set_tests_properties(${testName}_RestartFromFileT  PROPERTIES DEPENDS ${testName}_WriteRestartT)
  ENDIF()
ENDIF()
