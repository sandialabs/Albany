# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2D_taylorGreenVortex.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input2D_taylorGreenVortex.yaml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_2DTaylorGreenVortex_SDBC ${Albany.exe} input2D_taylorGreenVortex.yaml)
endif()

