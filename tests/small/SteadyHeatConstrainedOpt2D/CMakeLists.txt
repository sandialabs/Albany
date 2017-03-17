

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_dirichlet_dist_param.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_dirichlet_dist_param.xml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dirichlet_Dist_Param NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName} ${Albany.exe} input_dirichlet_dist_param.xml)
endif()

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_dirichlet_mixed_params.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_dirichlet_mixed_params.xml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dirichlet_Mixed_Params NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName} ${Albany.exe} input_dirichlet_mixed_params.xml)
endif()

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_conductivity_dist_param.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_conductivity_dist_param.xml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Conductivity_Dist_Param NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName} ${Albany.exe} input_conductivity_dist_param.xml)
endif()

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_conductivity_dist_param_restart.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_conductivity_dist_param_restart.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/steady2d_conduct_opt.exo
               ${CMAKE_CURRENT_BINARY_DIR}/steady2d_conduct_opt.exo COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Conductivity_Dist_Param_Restart NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName} ${Albany.exe} input_conductivity_dist_param_restart.xml)
endif()
