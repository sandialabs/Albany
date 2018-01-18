##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

if (ALBANY_IFPACK2)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TransientHeat2D.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/TransientHeat2D.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TransientHeat2D_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/TransientHeat2D_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ThermalExpansion.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/ThermalExpansion.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/LinearThermalExpansion.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/LinearThermalExpansion.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/LinearThermalExpansionMaterials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/LinearThermalExpansionMaterials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsWithTemperature.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsWithTemperature.yaml COPYONLY)
endif()

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if (ALBANY_IFPACK2)
    add_test(${testName}TransientHeat2D ${AlbanyT.exe} TransientHeat2D.yaml)
    add_test(${testName}LinearThermalExpansion ${AlbanyT.exe} LinearThermalExpansion.yaml)
endif()

# Thermal Expansion test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ThermalExpansion.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/ThermalExpansion.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ThermalExpansionMaterials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/ThermalExpansionMaterials.yaml COPYONLY)

# Coupled Plasticity
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsWithTemperature.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsWithTemperature.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsWithTemperatureMaterials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsWithTemperatureMaterials.yaml COPYONLY)
