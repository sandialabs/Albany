

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputBase.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputBase.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputBaseT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputBaseT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputCR.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputCR.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTM.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputTM.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/rip.exo
               ${CMAKE_CURRENT_BINARY_DIR}/rip.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName}_BaseProblem ${Albany.exe} inputBase.xml)
endif()
add_test(${testName}_BaseProblem_Tpetra ${AlbanyT.exe} inputBaseT.xml)
add_test(${testName}_CopyRemesh_Tpetra ${AlbanyT.exe} inputCR.xml)

IF (ALBANY_LCM)
#add_test(${testName}_TopMod ${Albany.exe} inputTM.xml)
ENDIF()
