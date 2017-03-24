# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputBase.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputBase.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputCR.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputCR.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTM.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputTM.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/rip.exo
               ${CMAKE_CURRENT_BINARY_DIR}/rip.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
add_test(${testName}_BaseProblem_Tpetra ${AlbanyT.exe} inputBase.yaml)
add_test(${testName}_CopyRemesh_Tpetra ${AlbanyT.exe} inputCR.yaml)

IF (ALBANY_LCM)
#add_test(${testName}_TopMod ${Albany.exe} inputTM.yaml)
ENDIF()
