# 1. Copy Input file from source to binary dir
CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/input_steady.xml
                ${CMAKE_CURRENT_BINARY_DIR}/input_steady.xml COPYONLY)
CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/input_potential_only.xml
                ${CMAKE_CURRENT_BINARY_DIR}/input_potential_only.xml COPYONLY)
#CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/input_unsteady.xml
#                ${CMAKE_CURRENT_BINARY_DIR}/input_unsteady.xml COPYONLY)

GET_FILENAME_COMPONENT (testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#2. Tests
IF (ALBANY_EPETRA)
#  ADD_TEST (${testName}_steady ${Albany.exe} input_steady.xml)
#  ADD_TEST (${testName}_unsteady ${Albany.exe} input_unsteady.xml)
ENDIF()
