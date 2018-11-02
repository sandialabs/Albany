if (ALBANY_IOPX)

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/domain.exo
               ${CMAKE_CURRENT_BINARY_DIR}/domain.exo COPYONLY)

# 3. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 4. Create the test (actually the data for the test) with this name and standard executable

if (ALBANY_EPETRA) 
add_test(${testName} ${Albany.exe} input.yaml)
endif()

if (ALBANY_IFPACK2)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)
add_test(${testName}_Tpetra_Rythmos ${AlbanyT.exe} inputT.yaml)
if (ALBANY_TEMPUS)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT_tempus.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT_tempus.yaml COPYONLY)
add_test(${testName}_Tpetra_Tempus ${AlbanyT.exe} inputT_tempus.yaml)
endif()
# 2. Copy mesh files from source to binary dir
endif()

endif(ALBANY_IOPX)
