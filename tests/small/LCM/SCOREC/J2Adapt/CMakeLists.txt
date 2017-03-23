#*****************************************************************//
#    Albany 3.0:  Copyright 2016 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//


# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/assoc.txt
               ${CMAKE_CURRENT_BINARY_DIR}/assoc.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputVerificationT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputVerificationT.yaml COPYONLY)

# 2. Name the test with the directory name, prepended by SCOREC_
get_filename_component(testRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(testName "SCOREC_${testRoot}")

# 3. Create the test with this name and standard executable
IF(ALBANY_IFPACK2)
  add_test(NAME ${testName}_Tpetra COMMAND ${AlbanyT.exe} inputT.yaml)
  add_test(NAME ${testName}_Verification_Tpetra COMMAND ${AlbanyT.exe} inputVerificationT.yaml)
ENDIF()
