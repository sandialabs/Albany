#*****************************************************************//
#    Albany 3.0:  Copyright 2016 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//


# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSerial.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSerial.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputRegressFail.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputRegressFail.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_non_man.xmt_txt
               ${CMAKE_CURRENT_BINARY_DIR}/test_non_man.xmt_txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coarse_cube_model.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/coarse_cube_model.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coarse_cube0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/coarse_cube0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coarse_cube1.smb
               ${CMAKE_CURRENT_BINARY_DIR}/coarse_cube1.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coarse_cube2.smb
               ${CMAKE_CURRENT_BINARY_DIR}/coarse_cube2.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coarse_cube3.smb
               ${CMAKE_CURRENT_BINARY_DIR}/coarse_cube3.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/serial_coarse_cube0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/serial_coarse_cube0.smb COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable

add_test(NAME ${testName}_Tpetra
   COMMAND ${CMAKE_COMMAND}  "-DTEST_PROG=${AlbanyT.exe}" -DTEST_ARGS=
   "-DFILENAME=coarse_cube_out.pvd" -P
   ${CMAKE_CURRENT_SOURCE_DIR}/pvdwritten.cmake)

IF(NOT ALBANY_PARALLEL_ONLY)
  add_test(NAME ${testName}_SERIAL_Tpetra
     COMMAND ${CMAKE_COMMAND}  "-DTEST_PROG=${SerialAlbanyT.exe}" -DTEST_ARGS=inputSerial.xml
     "-DFILENAME=serial_coarse_cube_out.pvd" -P
     ${CMAKE_CURRENT_SOURCE_DIR}/pvdwritten.cmake)
  add_test(NAME ${testName}_Tpetra_RegressFail
     COMMAND ${SerialAlbanyT.exe} inputRegressFail.xml)
  set_tests_properties(${testName}_Tpetra_RegressFail PROPERTIES WILL_FAIL TRUE)
ENDIF()
