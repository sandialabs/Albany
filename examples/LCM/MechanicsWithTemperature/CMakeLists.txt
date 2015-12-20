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

if (ALBANY_IFPACK2)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsTransientHeat2DT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsTransientHeat2DT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ThermalExpansionT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/ThermalExpansionT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/LinearThermalExpansionT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/LinearThermalExpansionT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/LinearThermalExpansionMaterials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/LinearThermalExpansionMaterials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsWithTemperatureT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsWithTemperatureT.xml COPYONLY)
endif()

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName}TransientHeat2D ${Albany.exe} MechanicsTransientHeat2D.xml)
endif()

if (ALBANY_IFPACK2) 
    add_test(${testName}TransientHeat2D_Tpetra ${AlbanyT.exe} MechanicsTransientHeat2DT.xml)
    add_test(${testName}LinearThermalExpansion_Tpetra ${AlbanyT.exe} LinearThermalExpansionT.xml)
endif()
if (ALBANY_MUELU_EXAMPLES)
  if (ALBANY_64BIT_INT)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsTransientHeat2DT_MueLu_64.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsTransientHeat2DT_MueLu_64.xml COPYONLY)
    add_test(${testName}TransientHeat2D_Tpetra_64 ${AlbanyT.exe} MechanicsTransientHeat2DT_MueLu_64.xml)
  else ()
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsTransientHeat2DT_MueLu.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsTransientHeat2DT_MueLu.xml COPYONLY)
    add_test(${testName}TransientHeat2D_Tpetra_MueLu ${AlbanyT.exe} MechanicsTransientHeat2DT_MueLu.xml)
  endif ()
endif()

# Thermal Expansion test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ThermalExpansion.xml
               ${CMAKE_CURRENT_BINARY_DIR}/ThermalExpansion.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ThermalExpansionMaterials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/ThermalExpansionMaterials.xml COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName}ThermalExpansion ${Albany.exe} ThermalExpansion.xml)
endif()

# Coupled Plasticity
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsWithTemperature.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsWithTemperature.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsWithTemperatureMaterials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsWithTemperatureMaterials.xml COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName} ${Albany.exe} MechanicsWithTemperature.xml)
endif()
