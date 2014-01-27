##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Transient Heat test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsTransientHeat2D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsTransientHeat2D.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materialsTransientHeat2D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materialsTransientHeat2D.xml COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
add_test(${testName}TransientHeat2D ${Albany.exe} MechanicsTransientHeat2D.xml)

# Thermal Expansion test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ThermalExpansion.xml
               ${CMAKE_CURRENT_BINARY_DIR}/ThermalExpansion.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ThermalExpansionMaterials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/ThermalExpansionMaterials.xml COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
add_test(${testName}ThermalExpansion ${Albany.exe} ThermalExpansion.xml)

# Coupled Plasticity
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsWithTemperature.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsWithTemperature.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsWithTemperatureMaterials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsWithTemperatureMaterials.xml COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
add_test(${testName} ${Albany.exe} MechanicsWithTemperature.xml)
