
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_exodiff_test.py
               ${CMAKE_CURRENT_BINARY_DIR}/run_exodiff_test.py COPYONLY)

add_subdirectory(TensileBarPD)
add_subdirectory(TensileBarPS)
add_subdirectory(TensileBarPDPS)
