# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_Enthalpy_test.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_Enthalpy_test.xml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
#add_test(${testName} ${SerialAlbany.exe} input_Enthalpy_test.xml)
