# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_FO_Thermo_test.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_FO_Thermo_test.xml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA AND ALBANY_IOPX)
add_test(${testName} ${Albany.exe} input_FO_Thermo_test.xml)
endif()
