
# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_steadyStokes2DT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_steadyStokes2DT.yaml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if (ALBANY_IFPACK2)
#get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3'. Create the test with this name and standard executable
add_test(${testName}_Steady_Stokes2D ${AlbanyT.exe} input_steadyStokes2DT.yaml)
endif ()
