##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Transient Heat test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsTransientHeat2D.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsTransientHeat2D.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materialsTransientHeat2D.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materialsTransientHeat2D.yaml COPYONLY)

if (ALBANY_IFPACK2)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsTransientHeat2DT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsTransientHeat2DT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ThermalExpansionT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/ThermalExpansionT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/LinearThermalExpansionT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/LinearThermalExpansionT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/LinearThermalExpansionMaterials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/LinearThermalExpansionMaterials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsWithTemperatureT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsWithTemperatureT.yaml COPYONLY)
endif()

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName}TransientHeat2D ${Albany.exe} MechanicsTransientHeat2D.yaml)
endif()

if (ALBANY_IFPACK2) 
    add_test(${testName}TransientHeat2D ${AlbanyT.exe} MechanicsTransientHeat2DT.yaml)
    add_test(${testName}LinearThermalExpansion ${AlbanyT.exe} LinearThermalExpansionT.yaml)
endif()
if (ALBANY_MUELU_EXAMPLES)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsTransientHeat2DT_MueLu.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsTransientHeat2DT_MueLu.yaml COPYONLY)
    add_test(${testName}TransientHeat2D_MueLu ${AlbanyT.exe} MechanicsTransientHeat2DT_MueLu.yaml)
endif()

# Thermal Expansion test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ThermalExpansion.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/ThermalExpansion.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ThermalExpansionMaterials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/ThermalExpansionMaterials.yaml COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName}ThermalExpansion ${Albany.exe} ThermalExpansion.yaml)
endif()

# Coupled Plasticity
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsWithTemperature.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsWithTemperature.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsWithTemperatureMaterials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsWithTemperatureMaterials.yaml COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName} ${Albany.exe} MechanicsWithTemperature.yaml)
endif()
