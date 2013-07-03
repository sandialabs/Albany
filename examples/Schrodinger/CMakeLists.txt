

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input1D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input1D.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and executable
add_test(${testName}_1D ${Albany.exe} input1D.xml)

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input2D.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/square2D.exo
               ${CMAKE_CURRENT_BINARY_DIR}/square2D.exo COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and executable
IF(NOT ALBANY_PARALLEL_ONLY)
  add_test(${testName}_2D_SERIAL ${SerialAlbany.exe} input2D.xml)
ENDIF()

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input3D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input3D.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube3D.exo
               ${CMAKE_CURRENT_BINARY_DIR}/cube3D.exo COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and executable
IF(NOT ALBANY_PARALLEL_ONLY)
  add_test(${testName}_3D_SERIAL ${SerialAlbany.exe} input3D.xml)
ENDIF()


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/infiniteWall1D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/infiniteWall1D.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
add_test(${testName}_infiniteWall1D ${Albany.exe} infiniteWall1D.xml)


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/infiniteWall2D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/infiniteWall2D.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
add_test(${testName}_infiniteWall2D ${Albany.exe} infiniteWall2D.xml)


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/finiteWall1D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/finiteWall1D.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
add_test(${testName}_finiteWall1D ${Albany.exe} finiteWall1D.xml)
