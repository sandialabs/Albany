

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTrans.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputTrans.xml COPYONLY)
# 2. Copy materials file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/qdot.dat
               ${CMAKE_CURRENT_BINARY_DIR}/qdot.dat COPYONLY)

# 3. Copy mesh files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ebs.exo
               ${CMAKE_CURRENT_BINARY_DIR}/ebs.exo COPYONLY)

# 5. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 6. Create the test (actually the data for the test) with this name and standard executable

if (ALBANY_EPETRA) 
add_test(${testName}_Steady ${SerialAlbany.exe})
# 7. Create the test with this name and standard executable
add_test(${testName} ${Albany.exe} inputTrans.xml)
set_tests_properties(${testName}  PROPERTIES DEPENDS ${testName}_Serial)
endif()
