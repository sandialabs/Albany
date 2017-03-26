##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Mechanics
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Mechanics.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/Mechanics.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsWithTemperature.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsWithTemperature.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsMaterials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsMaterials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsWithTemperatureMaterials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsWithTemperatureMaterials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/brick_model_2.g
               ${CMAKE_CURRENT_BINARY_DIR}/brick_model_2.g COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

IF (ALBANY_MUELU_EXAMPLES)
  add_test (${testName}_mech ${AlbanyT.exe} Mechanics.yaml)
  set_tests_properties(${testName}_mech PROPERTIES LABELS CUDA_TEST)
  add_test (${testName}_thermomech ${AlbanyT.exe} MechanicsWithTemperature.yaml)
  set_tests_properties(${testName}_thermomech PROPERTIES LABELS CUDA_TEST)
ENDIF ()
