# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/build_cism_msh_from_nc.m
               ${CMAKE_CURRENT_BINARY_DIR}/build_cism_msh_from_nc.m COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/compare_exos.m
               ${CMAKE_CURRENT_BINARY_DIR}/compare_exos.m COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/exo_read.m
               ${CMAKE_CURRENT_BINARY_DIR}/exo_read.m COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/extend_field.m
               ${CMAKE_CURRENT_BINARY_DIR}/extend_field.m COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nc_to_msh_cut_quad.m
               ${CMAKE_CURRENT_BINARY_DIR}/nc_to_msh_cut_quad.m COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/netcdf_read.m
               ${CMAKE_CURRENT_BINARY_DIR}/netcdf_read.m COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/print_exo_fields_into_nc.m 
               ${CMAKE_CURRENT_BINARY_DIR}/print_exo_fields_into_nc.m  COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/print_msh_quad.m 
               ${CMAKE_CURRENT_BINARY_DIR}/print_msh_quad.m  COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/README
               ${CMAKE_CURRENT_BINARY_DIR}/README COPYONLY)

