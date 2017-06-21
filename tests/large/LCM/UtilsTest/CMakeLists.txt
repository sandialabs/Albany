

# 1. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 2. Create the test with this name and standard executable
add_test(${testName} ${UtilsTest.exe})
