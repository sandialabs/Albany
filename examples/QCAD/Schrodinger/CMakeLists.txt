
# Copy files used by multiple tests
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/output)

# Get testRoot as QCAD_<the directory name>
get_filename_component(subRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(testRoot "QCAD_${subRoot}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_parabolic1D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_parabolic1D.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_parabolic2D_qcad.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_parabolic2D_qcad.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_parabolic3D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_parabolic3D.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_infiniteWall1D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_infiniteWall1D.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_infiniteWall2D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_infiniteWall2D.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_finiteWall1D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_finiteWall1D.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_formula2D_qcad.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_formula2D_qcad.xml COPYONLY)


# Create tests with this name and serial executable
IF(NOT ALBANY_PARALLEL_ONLY)
  add_test(${testRoot}_parabolic1D_SERIAL ${SerialAlbany.exe} input_parabolic1D.xml)	
  add_test(${testRoot}_parabolic2D_SERIAL ${SerialAlbany.exe} input_parabolic2D_qcad.xml)
ENDIF()

# Create tests with this name and standard executable
add_test(${testRoot}_parabolic1D ${Albany.exe} input_parabolic1D.xml)
add_test(${testRoot}_parabolic2D ${Albany.exe} input_parabolic2D_qcad.xml)
add_test(${testRoot}_parabolic3D ${Albany.exe} input_parabolic3D.xml)
add_test(${testRoot}_formula2D_qcad ${Albany.exe} input_formula2D_qcad.xml)
add_test(${testRoot}_infiniteWall1D ${Albany.exe} input_infiniteWall1D.xml)
add_test(${testRoot}_infiniteWall2D ${Albany.exe} input_infiniteWall2D.xml)
add_test(${testRoot}_finiteWall1D ${Albany.exe} input_finiteWall1D.xml)
