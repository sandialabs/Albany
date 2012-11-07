

# 1. Copy Input file from source to binary dir

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input2D.xml COPYONLY)
add_test(${testName}_2DSteady ${SerialAlbany.exe} input2D.xml)

# 4. Create the test with this name and standard executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2Dunsteady.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input2Dunsteady.xml COPYONLY)
add_test(${testName}_2DUnsteadyInvPressPulse ${SerialAlbany.exe} input2Dunsteady.xml)

# 5. Create the test with this name and standard executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2DunsteadyMMS.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input2DunsteadyMMS.xml COPYONLY)
add_test(${testName}_2DUnsteadyEulerMMS ${SerialAlbany.exe} input2DunsteadyMMS.xml)

# 5. Create the test with this name and standard executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input3Dunsteady.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input3Dunsteady.xml COPYONLY)
add_test(${testName}_3DUnsteadyInvPressPulse ${SerialAlbany.exe} input3Dunsteady.xml)

# 6. Create the test with this name and standard executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2DunsteadyNS.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input2DunsteadyNS.xml COPYONLY)
add_test(${testName}_2DUnsteadyInvPressPulseNS ${SerialAlbany.exe}
input3DunsteadyNS.xml)

# 7. Create the test with this name and standard executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input3DunsteadyNS.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input3DunsteadyNS.xml COPYONLY)
add_test(${testName}_3DUnsteadyInvPressPulseNS ${SerialAlbany.exe}
input3DunsteadyNS.xml)
