configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_advection_notopo.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_advection_notopo.xml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_advection_notopo_hv.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_advection_notopo_hv.xml COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

IF(ALBANY_KOKKOS_UNDER_DEVELOPMENT)
add_test(Aeras_${testName}_1_noHV  ${AlbanyT.exe} input_advection_notopo.xml)
ENDIF()
add_test(Aeras_${testName}_1_HV ${AlbanyT.exe} input_advection_notopo_hv.xml)




