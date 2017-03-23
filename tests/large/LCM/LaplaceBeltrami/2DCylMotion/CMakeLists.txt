
# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputDTPS.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputDTPS.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTPSA.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputTPSA.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputDLB.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputDLB.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cyl.exo
               ${CMAKE_CURRENT_BINARY_DIR}/cyl.exo COPYONLY)

# 4. Name the test with the directory name
get_filename_component(testDirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 5. Create the test with this name and standard executable
set(testName ${testDirName})

if (ALBANY_EPETRA) 
add_test(LaplaceBeltrami:${testName}_DTPS ${Albany.exe} inputDTPS.yaml)
endif()
