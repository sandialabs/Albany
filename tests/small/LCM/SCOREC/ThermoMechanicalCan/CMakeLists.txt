##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Mechanics
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsWithTemperatureT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsWithTemperatureT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsTimeDepTemperatureT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsTimeDepTemperatureT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsWithTemperatureMaterials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsWithTemperatureMaterials.yaml COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
if (ALBANY_MUELU_EXAMPLES)
  add_test (SCOREC_${testName}_mech_tpetra ${AlbanyT.exe} MechanicsT.yaml)
  set_tests_properties(SCOREC_${testName}_mech_tpetra PROPERTIES LABELS CUDA_TEST)
  add_test (SCOREC_${testName}_thermomech_tpetra ${AlbanyT.exe} MechanicsWithTemperatureT.yaml)
  set_tests_properties(SCOREC_${testName}_thermomech_tpetra PROPERTIES LABELS CUDA_TEST)
  add_test (SCOREC_${testName}_timedep_thermomech_tpetra ${AlbanyT.exe} MechanicsTimeDepTemperatureT.yaml)
  set_tests_properties(SCOREC_${testName}_timedep_thermomech_tpetra PROPERTIES LABELS CUDA_TEST)
ENDIF()
