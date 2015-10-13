# 1. Copy Input file from source to binary dir
CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/input_hewitt.xml
                ${CMAKE_CURRENT_BINARY_DIR}/input_hewitt.xml COPYONLY)

GET_FILENAME_COMPONENT (testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#2. Tests
IF (ALBANY_EPETRA)
  ADD_TEST (${testName}_hewitt ${Albany.exe} input_hewitt.xml)
ENDIF()
