##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Mechanics
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsWithTemperatureT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsWithTemperatureT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsWithTemperatureMaterials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsWithTemperatureMaterials.xml COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
# GAH - working this one now! Uncomment to run it
#IF(ENABLE_MESHMODEL)
#  IF(ALBANY_IFPACK2)
#    add_test (SCOREC_${testName}_thermomech_tpetra ${AlbanyT.exe} MechanicsWithTemperatureT.xml)
#    set_tests_properties(SCOREC_${testName}_thermomech_tpetra PROPERTIES LABELS CUDA_TEST)
#  ENDIF()
#ENDIF()

#if (ALBANY_MUELU_EXAMPLES)
#  add_test (${testName}_thermomech_tpetra_SCOREC ${AlbanyT.exe} MechanicsWithTemperatureT.xml)
#  set_tests_properties(${testName}_thermomech_tpetra_SCOREC PROPERTIES LABELS CUDA_TEST)
#endif ()

