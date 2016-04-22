configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_advection_notopo.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_advection_notopo.xml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_advection_notopo_hv.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_advection_notopo_hv.xml COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

add_test(Aeras_${testName}_Pure_Advection1_noHV  ${AlbanyT.exe} input_advection_notopo.xml)
add_test(Aeras_${testName}_Pure_Advection1_HV ${AlbanyT.exe} input_advection_notopo_hv.xml)




