configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_ne10_np4_hv5e15_quarter_day_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_ne10_np4_hv5e15_quarter_day_T.yaml COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

IF(NOT ALBANY_CHECK_FPE)
IF(NOT ALBANY_STRONG_FPE_CHECK) 
add_test(Aeras_${testName}_hv ${AlbanyT.exe} input_ne10_np4_hv5e15_quarter_day_T.yaml)
ENDIF()
ENDIF()



