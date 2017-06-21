#*****************************************************************//
#    Albany 3.0:  Copyright 2016 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAdapt.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputAdapt.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lshape.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/lshape.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lshape-serial0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/lshape-serial0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lshape-materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/lshape-materials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lshape-assoc.txt
               ${CMAKE_CURRENT_BINARY_DIR}/lshape-assoc.txt COPYONLY)

# 2. Name the test with the directory name, prepended by ANISO_
get_filename_component(testRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(testName "ANISO_${testRoot}")

# 3. Create the test with this name and standard executable
IF(ALBANY_IFPACK2)
  add_test(NAME ${testName}_Lshape COMMAND ${SerialAlbanyT.exe} inputAdapt.yaml)
ENDIF()
