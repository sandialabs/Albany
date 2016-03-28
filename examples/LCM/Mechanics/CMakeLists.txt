##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/2materials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2hex.g
               ${CMAKE_CURRENT_BINARY_DIR}/2hex.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/quad2d_tpetra_in.e.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/quad2d_tpetra_in.e.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/quad2d_tpetra_in.e.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/quad2d_tpetra_in.e.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/quad2d_tpetra_in.e.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/quad2d_tpetra_in.e.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/quad2d_tpetra_in.e.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/quad2d_tpetra_in.e.4.3 COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY)
  if (ALBANY_EPETRA)
    add_test(${testName}_SERIAL ${SerialAlbany.exe})
  endif()
#  IF(ALBANY_IFPACK2)
  # Teko needs Epetra - this now uses Teko preconditioning
  if (ALBANY_EPETRA)
    add_test(${testName}_Tpetra_SERIAL ${SerialAlbanyT.exe} inputT.xml)
  ENDIF()
ENDIF()

# test for 2D with blocked ordering
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputBlocked.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputBlocked.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputBlockedT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputBlockedT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
if (ALBANY_EPETRA)
add_test(${testName}2D_Blocked ${Albany.exe} inputBlocked.xml)
endif()
IF(ALBANY_IFPACK2)
  add_test(${testName}2D_Blocked_Tpetra ${AlbanyT.exe} inputBlockedT.xml)
ENDIF()

# test for 2D with J2 plasticity model
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputJ2Plasticity2D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputJ2Plasticity2D.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputJ2Plasticity2DT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputJ2Plasticity2DT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputJ2Plasticity2DT_dir_fieldBC.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputJ2Plasticity2DT_dir_fieldBC.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputJ2Plast2DTraction.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputJ2Plast2DTraction.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/J2.xml
               ${CMAKE_CURRENT_BINARY_DIR}/J2.xml COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
if (ALBANY_EPETRA)
add_test(${testName}2D_J2 ${Albany.exe} inputJ2Plasticity2D.xml)
endif()
IF(ALBANY_IFPACK2)
  add_test(${testName}2D_J2_Tpetra ${AlbanyT.exe} inputJ2Plasticity2DT.xml)
  add_test(${testName}2D_J2_dir_fieldBC_Tpetra ${AlbanyT.exe} inputJ2Plasticity2DT_dir_fieldBC.xml)
ENDIF()
if (ALBANY_EPETRA)
add_test(${testName}2D_J2_Trac ${Albany.exe} inputJ2Plast2DTraction.xml)
endif()
