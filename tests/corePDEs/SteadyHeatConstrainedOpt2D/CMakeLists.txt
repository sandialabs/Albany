# The root name for the tests in this directory is the directory name
get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${parentDirName}_${dirName})

##################################
### Dirichlet Dist Param tests ###
##################################

set(testName ${testNameRoot}_Dirichlet_Dist_Param)

# Copy Input files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_dirichlet_dist_paramT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_dirichlet_dist_paramT.yaml COPYONLY)

# Create the test with this name and standard executable
add_test(${testName}_Tpetra ${Albany.exe} input_dirichlet_dist_paramT.yaml)
set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "Basic;Tpetra;Forward")

if (ALBANY_EPETRA) 
  # Copy Input files from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_dirichlet_dist_param.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_dirichlet_dist_param.yaml COPYONLY)

  # Create the test with this name and standard executable
  add_test(${testName}_Epetra ${Albany.exe} input_dirichlet_dist_param.yaml)
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Basic;Epetra;Forward")
endif()

if (ALBANY_ROL)
  add_test(${testName}_Analysis ${AlbanyAnalysis.exe} input_dirichlet_dist_paramT.yaml)
  set_tests_properties(${testName}_Analysis PROPERTIES LABELS "Basic;Tpetra;Analysis;ROL")
endif ()

####################################
### Dirichlet Mixed Params tests ###
####################################

set(testName ${testNameRoot}_Dirichlet_Mixed_Params)

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_dirichlet_mixed_paramsT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_dirichlet_mixed_paramsT.yaml COPYONLY)

# Create the test with this name and standard executable
add_test(${testName}_Tpetra ${Albany.exe} input_dirichlet_mixed_paramsT.yaml)
set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "Basic;Tpetra;Forward")

if (ALBANY_EPETRA) 
  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_dirichlet_mixed_params.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_dirichlet_mixed_params.yaml COPYONLY)

  # Create the test with this name and standard executable
  add_test(${testName}_Epetra ${Albany.exe} input_dirichlet_mixed_params.yaml)
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Basic;Epetra;Forward")
endif()

set(testName ${testNameRoot}_Dirichlet_Mixed_Params_Analysis)

if (ALBANY_ROL)
  add_test(${testName}_Tpetra ${AlbanyAnalysis.exe} input_dirichlet_mixed_paramsT.yaml)
  set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "Basic;Tpetra;Analysis;ROL")
if (ALBANY_EPETRA)
  add_test(${testName}_Epetra ${AlbanyAnalysis.exe} input_dirichlet_mixed_params.yaml)
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Basic;Epetra;Analysis;ROL")
endif() 
endif()

####################################
### Distributed Params tests ###
####################################

set(testName ${testNameRoot}_Conductivity_Dist_Param)

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_conductivity_dist_paramT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_conductivity_dist_paramT.yaml COPYONLY)
# Create the test with this name and standard executable
add_test(${testName}_Tpetra ${Albany.exe} input_conductivity_dist_paramT.yaml)
set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "Basic;Tpetra;Forward")

if (ALBANY_PARALELL_EXODUS AND NOT ALBANY_DISABLE_OUTPUT_MESH)
  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_conductivity_dist_param_restartT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_conductivity_dist_param_restartT.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/steady2d_conduct_opt.exo
                 ${CMAKE_CURRENT_BINARY_DIR}/steady2d_conduct_opt.exo COPYONLY)

  # Create the test with this name and standard executable
  add_test(${testName}_Restart_Tpetra ${Albany.exe} input_conductivity_dist_param_restartT.yaml)
  set_tests_properties(${testName}_Restart_Tpetra PROPERTIES LABELS "Basic;Tpetra;Forward")
  set_tests_properties(${testName}_Restart_Tpetra PROPERTIES DEPENDS  ${testName}_Tpetra)
endif()

if (ALBANY_EPETRA) 
  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_conductivity_dist_param.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_conductivity_dist_param.yaml COPYONLY)
  # Create the test with this name and standard executable
  add_test(${testName}_Epetra ${Albany.exe} input_conductivity_dist_param.yaml)
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Basic;Epetra;Forward")

  if (ALBANY_PARALELL_EXODUS AND NOT ALBANY_DISABLE_OUTPUT_MESH)
    # Copy Input file from source to binary dir
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_conductivity_dist_param_restart.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_conductivity_dist_param_restart.yaml COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/steady2d_conduct_opt.exo
                   ${CMAKE_CURRENT_BINARY_DIR}/steady2d_conduct_opt.exo COPYONLY)

    # Create the test with this name and standard executable
    add_test(${testName}_Restart_Epetra ${Albany.exe} input_conductivity_dist_param_restart.yaml)
    set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Basic;Epetra;Forward")
    set_tests_properties(${testName}_Restart_Epetra PROPERTIES DEPENDS  ${testName}_Epetra)
  endif()
endif()


####################################
### Mixed Scalar Distributed Params tests ###
####################################

set(testName ${testNameRoot}_Scalar_And_Dist_Param)

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_scalar_and_dist_paramT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_scalar_and_dist_paramT.yaml COPYONLY)
# Create the test with this name and standard executable
add_test(${testName}_Tpetra ${Albany.exe} input_scalar_and_dist_paramT.yaml)
set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "Basic;Tpetra;Forward")

####################################
### Hessian Mixed Scalar Distributed Params tests ###
####################################

if (ALBANY_ROL AND ALBANY_TEKO)
  set(testName ${testNameRoot}_Scalar_And_Dist_Param_Hessian)

  # Copy Input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_scalar_and_dist_param_hessianT.yaml
                ${CMAKE_CURRENT_BINARY_DIR}/input_scalar_and_dist_param_hessianT.yaml COPYONLY)
  # Create the test with this name and standard executable
  add_test(${testName}_Tpetra ${AlbanyAnalysis.exe} input_scalar_and_dist_param_hessianT.yaml)
  set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "Basic;Tpetra;Analysis")
endif()
