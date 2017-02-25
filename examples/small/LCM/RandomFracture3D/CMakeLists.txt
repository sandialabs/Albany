

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputRandom.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputRandom.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_s_Random.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_s_Random.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable

#IF (ALBANY_LCM)
#add_test(${testName} ${Albany.exe} inputRandom.xml)
#ENDIF()
