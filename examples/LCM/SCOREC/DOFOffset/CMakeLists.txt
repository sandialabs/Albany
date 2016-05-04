#*****************************************************************//
#    Albany 3.0:  Copyright 2016 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//


# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputIfpack2.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputIfpack2.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMueLu.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMueLu.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/assoc.txt
               ${CMAKE_CURRENT_BINARY_DIR}/assoc.txt COPYONLY)

# 2. Name the test with the directory name, prepended by SCOREC_
get_filename_component(testRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(testName "SCOREC_${testRoot}")

# 3. Create the test with this name and standard executable
# Only run tests in 64 bit mode (they should fail everywhere else ??)
IF (ALBANY_64BIT_INT AND NOT ALBANY_EPETRA)
  IF(ENABLE_MESHMODEL)
    IF(ALBANY_IFPACK2)
      add_test(NAME ${testName}_Ifpack2_Tpetra COMMAND ${AlbanyT.exe} inputIfpack2.xml)
    ENDIF()
    IF(ALBANY_MUELU_EXAMPLES)
      add_test(NAME ${testName}_MueLu_Tpetra COMMAND ${SerialAlbanyT.exe} inputMueLu.xml)
    ENDIF()
  ENDIF()
ENDIF()
