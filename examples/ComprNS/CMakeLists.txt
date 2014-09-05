
# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2D_cavity.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input2D_cavity.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cavl2soln.exo
               ${CMAKE_CURRENT_BINARY_DIR}/cavl2soln.exo COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2D_taylorGreenVortex.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input2D_taylorGreenVortex.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_2DTaylorGreenVortex ${Albany.exe} input2D_taylorGreenVortex.xml)
endif()

