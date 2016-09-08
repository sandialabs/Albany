##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-0.g
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-0.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-1.g
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-1.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials-0.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials-0.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials-1.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials-1.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-0.xml
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-0.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-1.xml
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-1.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-S.xml
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-S.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-S-ctest.xml
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-S-ctest.xml COPYONLY)

if (ALBANY_DTK)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtestT_dtk_8proc.py
               ${CMAKE_CURRENT_BINARY_DIR}/runtestT_dtk_8proc.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-0.g.8.0
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-0.g.8.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-0.g.8.1
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-0.g.8.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-0.g.8.2
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-0.g.8.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-0.g.8.3
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-0.g.8.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-0.g.8.4
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-0.g.8.4 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-0.g.8.5
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-0.g.8.5 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-0.g.8.6
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-0.g.8.6 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-0.g.8.7
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-0.g.8.7 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-1.g.8.0
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-1.g.8.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-1.g.8.1
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-1.g.8.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-1.g.8.2
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-1.g.8.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-1.g.8.3
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-1.g.8.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-1.g.8.4
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-1.g.8.4 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-1.g.8.5
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-1.g.8.5 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-1.g.8.6
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-1.g.8.6 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/notched-cylinder-1.g.8.7
               ${CMAKE_CURRENT_BINARY_DIR}/notched-cylinder-1.g.8.7 COPYONLY)
endif()
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyTPath} ${CMAKE_CURRENT_BINARY_DIR}/AlbanyT)
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_DTK) 
  add_test(NAME Schwarz_NotchedCylinder_${testName}_DTK_8proc COMMAND "python" "runtestT_dtk_8proc.py")
endif()
