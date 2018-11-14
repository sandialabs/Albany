

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_dirichlet_dist_paramT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_dirichlet_dist_paramT.yaml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dirichlet_Dist_ParamT NAME)
# 3. Create the test with this name and standard executable
add_test(${testName} ${AlbanyT.exe} input_dirichlet_dist_paramT.yaml)

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_dirichlet_dist_param.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_dirichlet_dist_param.yaml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dirichlet_Dist_Param NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName} ${Albany.exe} input_dirichlet_dist_param.yaml)
endif()

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_dirichlet_mixed_paramsT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_dirichlet_mixed_paramsT.yaml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dirichlet_Mixed_ParamsT NAME)
# 3. Create the test with this name and standard executable
add_test(${testName} ${AlbanyT.exe} input_dirichlet_mixed_paramsT.yaml)

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_dirichlet_mixed_params.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_dirichlet_mixed_params.yaml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dirichlet_Mixed_Params NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName} ${Albany.exe} input_dirichlet_mixed_params.yaml)
endif()

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_conductivity_dist_param.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_conductivity_dist_param.yaml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Conductivity_Dist_Param NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName} ${Albany.exe} input_conductivity_dist_param.yaml)
endif()

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_conductivity_dist_paramT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_conductivity_dist_paramT.yaml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Conductivity_Dist_ParamT NAME)
# 3. Create the test with this name and standard executable
add_test(${testName} ${AlbanyT.exe} input_conductivity_dist_paramT.yaml)

if (ALBANY_IOPX)
# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_conductivity_dist_param_restart.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_conductivity_dist_param_restart.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/steady2d_conduct_opt.exo
               ${CMAKE_CURRENT_BINARY_DIR}/steady2d_conduct_opt.exo COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Conductivity_Dist_Param_Restart NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName} ${Albany.exe} input_conductivity_dist_param_restart.yaml)
endif()

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_conductivity_dist_param_restartT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_conductivity_dist_param_restartT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/steady2d_conduct_opt.exo
               ${CMAKE_CURRENT_BINARY_DIR}/steady2d_conduct_opt.exo COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Conductivity_Dist_Param_RestartT NAME)
# 3. Create the test with this name and standard executable
add_test(${testName} ${AlbanyT.exe} input_conductivity_dist_param_restartT.yaml)
endif()
