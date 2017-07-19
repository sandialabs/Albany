# 1. Copy Input file from source to binary dir
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/input_2D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_2D.xml COPYONLY)
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/input_3D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_3D.xml COPYONLY)

# 2. Name the test with the directory name
GET_FILENAME_COMPONENT(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#3. Tests
ADD_TEST(${testName}_2D ${Albany.exe} input_2D.xml)
ADD_TEST(${testName}_3D ${Albany.exe} input_3D.xml)
