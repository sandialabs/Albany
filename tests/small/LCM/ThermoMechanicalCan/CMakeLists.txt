##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Mechanics
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Mechanics.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/Mechanics.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsWithTemperatureT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsWithTemperatureT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsMaterials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsMaterials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsWithTemperatureMaterials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsWithTemperatureMaterials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/brick_model_2.g
               ${CMAKE_CURRENT_BINARY_DIR}/brick_model_2.g COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
if (ALBANY_EPETRA)
  add_test(${testName}_mech_epetra ${Albany.exe} Mechanics.yaml)
endif ()
if (ALBANY_MUELU_EXAMPLES)
  add_test (${testName}_mech_tpetra ${AlbanyT.exe} MechanicsT.yaml)
  set_tests_properties(${testName}_mech_tpetra PROPERTIES LABELS CUDA_TEST)
  add_test (${testName}_thermomech_tpetra ${AlbanyT.exe} MechanicsWithTemperatureT.yaml)
  set_tests_properties(${testName}_thermomech_tpetra PROPERTIES LABELS CUDA_TEST)
endif ()
