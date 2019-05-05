configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_advection_notopo.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_advection_notopo.yaml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_advection_notopo_hv.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_advection_notopo_hv.yaml COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

IF(ALBANY_KOKKOS_UNDER_DEVELOPMENT)
add_test(Aeras_${testName}_1_noHV  ${Albany.exe} input_advection_notopo.yaml)
set_tests_properties(Aeras_${testName}_1_noHV PROPERTIES LABELS "Aeras;Tpetra;Forward")
ENDIF()
add_test(Aeras_${testName}_1_HV ${Albany.exe} input_advection_notopo_hv.yaml)
set_tests_properties(Aeras_${testName}_1_HV PROPERTIES LABELS "Aeras;Tpetra;Forward")




