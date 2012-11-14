

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
# 2. Copy materials file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)

# 3. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 5. Create the test (actually the data for the test) with this name and standard executable

add_executable(HeatProfile HeatProfile.C)

add_test(${testName}_Prep HeatProfile)

# 5. Create the test with this name and standard executable

add_test(${testName} ${Albany.exe})
set_tests_properties(${testName}  PROPERTIES DEPENDS ${testName}_Prep)
