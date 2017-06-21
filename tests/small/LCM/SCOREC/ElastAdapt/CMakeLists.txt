#*****************************************************************//
#    Albany 3.0:  Copyright 2016 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//


# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSprT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSprT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSprT_postParma.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSprT_postParma.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSprT_postZoltan.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSprT_postZoltan.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputNeckingSerialT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputNeckingSerialT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputNeckingT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputNeckingT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTwistT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputTwistT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputShearT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputShearT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)

# 2. Name the test with the directory name, prepended by SCOREC_
get_filename_component(testRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(testName "SCOREC_${testRoot}")

# 3. Create the test with this name and standard executable
IF(ALBANY_IFPACK2)
  add_test(NAME ${testName}_SPR_Tpetra COMMAND ${AlbanyT.exe} inputSprT.yaml)
  add_test(NAME ${testName}_SPR_Tpetra_postParma COMMAND ${AlbanyT.exe} inputSprT_postParma.yaml)
  add_test(NAME ${testName}_SPR_Tpetra_postZoltan COMMAND ${AlbanyT.exe} inputSprT_postZoltan.yaml)
  add_test(NAME ${testName}_Necking_SERIAL_Tpetra COMMAND ${SerialAlbanyT.exe} inputNeckingSerialT.yaml)
  add_test(NAME ${testName}_Necking_Tpetra COMMAND ${AlbanyT.exe} inputNeckingT.yaml)
  add_test(NAME ${testName}_Twist_Tpetra COMMAND ${SerialAlbanyT.exe} inputTwistT.yaml)
#  add_test(NAME ${testName}_Shear_Tpetra COMMAND ${AlbanyT.exe} inputShearT.yaml)
ENDIF()
