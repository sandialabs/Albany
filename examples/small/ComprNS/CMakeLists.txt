# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2D_taylorGreenVortex.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input2D_taylorGreenVortex.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_2DTaylorGreenVortex ${Albany.exe} input2D_taylorGreenVortex.xml)
endif()

