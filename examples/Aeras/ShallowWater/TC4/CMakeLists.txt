 

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../grids/QUAD9/uniform_16_quad9.g
               ${CMAKE_CURRENT_BINARY_DIR}/uniform_16_quad9.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tc4_interp.ncl
               ${CMAKE_CURRENT_BINARY_DIR}/tc4_interp.ncl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tc4_plot_analyt_soln.ncl
               ${CMAKE_CURRENT_BINARY_DIR}/tc4_plot_analyt_soln.ncl COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
add_test(Aeras_${testName}_Serial ${SerialAlbany.exe})
#add_test(Aeras_${testName} ${Albany.exe})

