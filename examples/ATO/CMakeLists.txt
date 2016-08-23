##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

IF(NOT ALBANY_64BIT_INT)
add_subdirectory(DistParam_2D)
add_subdirectory(RegHeaviside_2D)
add_subdirectory(RegHeaviside_3D)
add_subdirectory(Mitchell_2D)
add_subdirectory(Mitchell_3D)
add_subdirectory(MultiLoadSet)
add_subdirectory(Filtered_2D)
add_subdirectory(Thermal_Mech_2D)
add_subdirectory(Thermal_MultiMech_3D)
add_subdirectory(FixedBlocks)
add_subdirectory(Restart)
add_subdirectory(Homogenize_2D)
add_subdirectory(MultiPhys_Homogenize_2D)
add_subdirectory(Constraint_2D)
add_subdirectory(CurrentOpt)
add_subdirectory(VolumeObj_StiffnessCon)
add_subdirectory(VolumeObj_StressCon)
add_subdirectory(2Matl_Homog)
add_subdirectory(BodyForce_2D)
add_subdirectory(ResidualStrain)
IF(ENABLE_Cogent)
add_subdirectory(Cogent_Tab)
ENDIF()
ENDIF()
