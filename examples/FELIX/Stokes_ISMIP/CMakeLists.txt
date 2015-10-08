
# 1. Copy Input file from source to binary dir

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#3. Tests
#  First test 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_ismip-hom_testA.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_ismip-hom_testA.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_TestA ${Albany.exe} input_ismip-hom_testA.xml)
endif()

