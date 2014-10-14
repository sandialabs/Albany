 

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../grids/QUAD9/uniform_16_quad9.g
               ${CMAKE_CURRENT_BINARY_DIR}/uniform_16_quad9.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/REF0007.cdf
               ${CMAKE_CURRENT_BINARY_DIR}/REF0007.cdf COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/grid_and_IC_16_quad9_dummy.nc
               ${CMAKE_CURRENT_BINARY_DIR}/grid_and_IC_16_quad9_dummy.nc COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/grid_and_IC_16_quad9.g
               ${CMAKE_CURRENT_BINARY_DIR}/grid_and_IC_16_quad9.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tc7_errorplot.ncl
               ${CMAKE_CURRENT_BINARY_DIR}/tc7_errorplot.ncl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tc7_plot_init_conditions.ncl
               ${CMAKE_CURRENT_BINARY_DIR}/tc7_plot_init_conditions.ncl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tc7_write_IC.ncl
               ${CMAKE_CURRENT_BINARY_DIR}/tc7_write_IC.ncl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/README
               ${CMAKE_CURRENT_BINARY_DIR}/README COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
add_test(Aeras_${testName}_Serial ${SerialAlbany.exe})

