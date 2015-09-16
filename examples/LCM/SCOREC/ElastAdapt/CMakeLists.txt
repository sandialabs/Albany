#*****************************************************************//
#    Albany 2.0:  Copyright 2012 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//


# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSprT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSprT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSprT_postParma.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSprT_postParma.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSprT_postZoltan.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSprT_postZoltan.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputNeckingSerialT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputNeckingSerialT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputNeckingT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputNeckingT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTwistT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputTwistT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputShearT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputShearT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)

# 2. Name the test with the directory name, prepended by SCOREC_
get_filename_component(testRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(testName "SCOREC_${testRoot}")

# 3. Create the test with this name and standard executable
IF(ENABLE_MESHMODEL)
  IF(ALBANY_IFPACK2)
    add_test(NAME ${testName}_SPR_Tpetra COMMAND ${AlbanyT.exe} inputSprT.xml)
    add_test(NAME ${testName}_SPR_Tpetra_postParma COMMAND ${AlbanyT.exe} inputSprT_postParma.xml)
    add_test(NAME ${testName}_SPR_Tpetra_postZoltan COMMAND ${AlbanyT.exe} inputSprT_postZoltan.xml)
    add_test(NAME ${testName}_Necking_SERIAL_Tpetra COMMAND ${SerialAlbanyT.exe} inputNeckingSerialT.xml)
    add_test(NAME ${testName}_Necking_Tpetra COMMAND ${AlbanyT.exe} inputNeckingT.xml)
#    add_test(NAME ${testName}_Shear_Tpetra COMMAND ${AlbanyT.exe} inputShearT.xml)
  ENDIF()
ENDIF()
