
# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_steadyStokes2DT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_steadyStokes2DT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_unsteadyNS2DT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_unsteadyNS2DT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_unsteadyNS2DT_with_fields.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_unsteadyNS2DT_with_fields.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vortex.exo
               ${CMAKE_CURRENT_BINARY_DIR}/vortex.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vortex.exo.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/vortex.exo.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vortex.exo.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/vortex.exo.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vortex.exo.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/vortex.exo.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vortex.exo.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/vortex.exo.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vortex_with_fields.exo
               ${CMAKE_CURRENT_BINARY_DIR}/vortex_with_fields.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vortex_with_fields.exo.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/vortex_with_fields.exo.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vortex_with_fields.exo.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/vortex_with_fields.exo.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vortex_with_fields.exo.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/vortex_with_fields.exo.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vortex_with_fields.exo.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/vortex_with_fields.exo.4.3 COPYONLY)
if (ALBANY_TEMPUS)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_boussinesq2D.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_boussinesq2D.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_boussinesq1D_solitary_wave.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_boussinesq1D_solitary_wave.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_boussinesq1D_generate_exo_with_fields.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_boussinesq1D_generate_exo_with_fields.yaml COPYONLY)
endif()
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if (ALBANY_MUELU)
#get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3'. Create the test with this name and standard executable
add_test(${testName}_Steady_Stokes2D ${AlbanyT.exe} input_steadyStokes2DT.yaml)
endif ()
if (ALBANY_IFPACK2)
add_test(${testName}_2D_Unsteady_Vortex_Shedding ${AlbanyT.exe} input_unsteadyNS2DT.yaml)
add_test(${testName}_2D_Unsteady_Vortex_Shedding_with_Fields ${AlbanyT.exe} input_unsteadyNS2DT_with_fields.yaml)
endif ()
