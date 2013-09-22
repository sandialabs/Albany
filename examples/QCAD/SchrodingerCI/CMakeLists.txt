
# Copy files used by multiple tests
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/output)

# Get testRoot as QCAD_<the directory name>
get_filename_component(subRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(testRoot "QCAD_${subRoot}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_parabolic2D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_parabolic2D.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_parabolic3D_um.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_parabolic3D_um.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_parabolic3D_nm.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_parabolic3D_nm.xml COPYONLY)

# Create tests with this name and standard executable
add_test(${testRoot}_parabolic2D ${Albany.exe} input_parabolic2D.xml)

#Removed these two since they're way too sensitive to numerics... like MPI vs non-MPI
#add_test(${testRoot}_parabolic3D_um ${Albany.exe} input_parabolic3D_um.xml)
#add_test(${testRoot}_parabolic3D_nm ${Albany.exe} input_parabolic3D_nm.xml)
