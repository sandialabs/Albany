get_filename_component(sub_root ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(test_root "SEE_${sub_root}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_stk_2D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_stk_2D.xml COPYONLY)

add_test(${test_root}_stk_2D ${AlbanyT.exe} input_stk_2D.xml)
