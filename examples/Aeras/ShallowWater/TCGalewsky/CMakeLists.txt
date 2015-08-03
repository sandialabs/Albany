 

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../grids/QUAD4/uniform_16_quad4.g
               ${CMAKE_CURRENT_BINARY_DIR}/uniform_16_quad4.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/regrid_and_plot_vort.ncl
               ${CMAKE_CURRENT_BINARY_DIR}/regrid_and_plot_vort.ncl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/plot_vort.ncl
               ${CMAKE_CURRENT_BINARY_DIR}/plot_vort.ncl COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
if (NOT ALBANY_MOR) 
add_test(Aeras_${testName}_Tpetra ${AlbanyT.exe} inputT.xml)
endif()
