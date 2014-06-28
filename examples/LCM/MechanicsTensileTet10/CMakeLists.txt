##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/NotchedTensileTet10.xml
               ${CMAKE_CURRENT_BINARY_DIR}/NotchedTensileTet10.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched_tensile_tet10.g
               ${CMAKE_CURRENT_BINARY_DIR}/notched_tensile_tet10.g COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
IF(ALBANY_SEACAS)
  add_test(${testName}_STK ${Albany.exe} NotchedTensileTet10.xml)
ENDIF()

#IF(ALBANY_SCOREC)
#  add_test(${testName}_FMDB ${Albany.exe} NotchedTensileTet10FMDB.xml)
#ENDIF()
