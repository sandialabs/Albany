#*****************************************************************//
#    Albany 3.0:  Copyright 2016 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
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

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
add_test(NAME ${testName}_Tpetra
         COMMAND ${CMAKE_COMMAND}  "-DTEST_PROG=${Albany.exe}" -DTEST_ARGS=input.yaml
         "-DFILENAME=coarse_cube_out.pvd" -P
         ${CMAKE_CURRENT_SOURCE_DIR}/pvdwritten.cmake)
set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "SCOREC;Tpetra;Forward")

IF(NOT ALBANY_PARALLEL_ONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSerial.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputSerial.yaml COPYONLY)
  add_test(NAME ${testName}_SERIAL_Tpetra
           COMMAND ${CMAKE_COMMAND}  "-DTEST_PROG=${SerialAlbany.exe}" -DTEST_ARGS=inputSerial.yaml
           "-DFILENAME=serial_coarse_cube_out.pvd" -P
           ${CMAKE_CURRENT_SOURCE_DIR}/pvdwritten.cmake)
  set_tests_properties(${testName}_SERIAL_Tpetra PROPERTIES LABELS "SCOREC;Tpetra;Forward")

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputRegressFail.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputRegressFail.yaml COPYONLY)
  add_test(NAME ${testName}_Tpetra_RegressFail
           COMMAND ${SerialAlbany.exe} inputRegressFail.yaml)
  set_tests_properties(${testName}_Tpetra_RegressFail PROPERTIES WILL_FAIL TRUE)
  set_tests_properties(${testName}_Tpetra_RegressFail PROPERTIES LABELS "SCOREC;Tpetra;Forward")
ENDIF()
