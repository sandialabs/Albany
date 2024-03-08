# 1. Copy Input file from source to binary dir
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/input_2D.yaml
                ${CMAKE_CURRENT_BINARY_DIR}/input_2D.yaml COPYONLY)
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/input_3D.yaml
                ${CMAKE_CURRENT_BINARY_DIR}/input_3D.yaml COPYONLY)

get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testName ${parentDirName}_${dirName})

ADD_TEST(${testName}_2D ${Albany.exe} input_2D.yaml)
SET_TESTS_PROPERTIES(${testName}_2D PROPERTIES LABELS "Basic;Forward")

ADD_TEST(${testName}_3D ${Albany.exe} input_3D.yaml)
SET_TESTS_PROPERTIES(${testName}_3D PROPERTIES LABELS "Basic;Forward")
