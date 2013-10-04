##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/2materials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2hex.g
               ${CMAKE_CURRENT_BINARY_DIR}/2hex.g COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY)
  add_test(${testName}_SERIAL ${SerialAlbany.exe})
ENDIF()

# test for 2D with blocked ordering
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputBlocked.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputBlocked.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
add_test(${testName}2D_Blocked ${Albany.exe} inputBlocked.xml)

# test for 2D with J2 plasticity model
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputJ2Plasticity2D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputJ2Plasticity2D.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputJ2Plast2DTraction.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputJ2Plast2DTraction.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/J2.xml
               ${CMAKE_CURRENT_BINARY_DIR}/J2.xml COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
add_test(${testName}2D_J2 ${Albany.exe} inputJ2Plasticity2D.xml)
add_test(${testName}2D_J2_Trac ${Albany.exe} inputJ2Plast2DTraction.xml)
