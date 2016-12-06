 

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.xml COPYONLY)
if (ALBANY_TEMPUS) 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTempus.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputTempus.xml COPYONLY)
endif () 
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tc3_interp.ncl
#               ${CMAKE_CURRENT_BINARY_DIR}/tc3_interp.ncl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tc3_native.ncl
               ${CMAKE_CURRENT_BINARY_DIR}/tc3_native.ncl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tc3_native_satellite.ncl
               ${CMAKE_CURRENT_BINARY_DIR}/tc3_native_satellite.ncl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../grids/QUAD4/uniform_16_quad4.g
               ${CMAKE_CURRENT_BINARY_DIR}/uniform_16_quad4.g COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
add_test(Aeras_${testName}_RungeKutta4 ${AlbanyT.exe} inputT.xml)
