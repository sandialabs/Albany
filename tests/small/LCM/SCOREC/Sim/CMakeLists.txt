#*****************************************************************//
#    Albany 3.0:  Copyright 2016 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//

# 0. If AMP is not enabled, bail
if (NOT ALBANY_AMP)
  return()
endif()

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAdapt.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputAdapt.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/annular.smd
               ${CMAKE_CURRENT_BINARY_DIR}/annular.smd COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/annular.sms
               ${CMAKE_CURRENT_BINARY_DIR}/annular.sms COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/annular_nat.x_t
               ${CMAKE_CURRENT_BINARY_DIR}/annular_nat.x_t COPYONLY)

# 2. Name the test with the directory name, prepended by SCOREC_
get_filename_component(testRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(testName "SCOREC_${testRoot}")

# 3. Create the test with this name and standard executable
IF(ALBANY_IFPACK2)
  add_test(NAME ${testName}_Tpetra COMMAND ${SerialAlbanyT.exe} input.yaml)
  add_test(NAME ${testName}_Adapt_Tpetra COMMAND ${SerialAlbanyT.exe} inputAdapt.yaml)
ENDIF()
