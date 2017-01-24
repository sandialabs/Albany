

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/domain.exo
               ${CMAKE_CURRENT_BINARY_DIR}/domain.exo COPYONLY)

# 3. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 4. Create the test (actually the data for the test) with this name and standard executable

if (ALBANY_EPETRA) 
add_test(${testName} ${Albany.exe})
endif()

if (ALBANY_IFPACK2)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.xml COPYONLY)
add_test(${testName}_Tpetra_Rythmos ${AlbanyT.exe} inputT.xml)
if (ALBANY_TEMPUS)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT_tempus.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT_tempus.xml COPYONLY)
add_test(${testName}_Tpetra_Tempus ${AlbanyT.exe} inputT_tempus.xml)
endif()
# 2. Copy mesh files from source to binary dir
endif()

