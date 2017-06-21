# 0. Set a variable that we can use to locate the catalyst pipeline script:
set(PIPELINE_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/pipeline.py")

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cyl.exo
               ${CMAKE_CURRENT_BINARY_DIR}/cyl.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pipeline.py
               ${CMAKE_CURRENT_BINARY_DIR}/pipeline.py COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable

if (ALBANY_IFPACK2) 
add_test(${testName} ${AlbanyT.exe} input.yaml)
endif()
