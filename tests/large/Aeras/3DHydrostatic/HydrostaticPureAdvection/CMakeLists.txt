configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_advection_notopo.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_advection_notopo.yaml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_advection_notopo_hv.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_advection_notopo_hv.yaml COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

IF(ALBANY_KOKKOS_UNDER_DEVELOPMENT)
add_test(Aeras_${testName}_1_noHV  ${AlbanyT.exe} input_advection_notopo.yaml)
ENDIF()
add_test(Aeras_${testName}_1_HV ${AlbanyT.exe} input_advection_notopo_hv.yaml)




