##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

MESSAGE ("\n ***********************")
MESSAGE (" *** LandIce Options ***")
MESSAGE (" ***********************\n")

# Set optional build of MPAS Interface (Land Ice SciDAC project) and CISM Interface
# Both OFF by default
option(ENABLE_CISM_INTERFACE "Flag to turn on LandIce Interface to the CISM code" OFF)
option(ENABLE_MPAS_INTERFACE "Flag to turn on LandIce Source code" OFF)
if (ENABLE_MPAS_INTERFACE)
  message("  -- MPAS_INTERFACE                        Enabled.")
else()
  message("  -- MPAS_INTERFACE                    NOT Enabled.")
endif()

if (ENABLE_CISM_INTERFACE)
  message("  -- CISM_INTERFACE                        Enabled.")
else()
  message("  -- CISM_INTERFACE                    NOT Enabled.")
endif()

set(SOURCES
  evaluators/LandIce_BasalFrictionCoefficient.cpp
  evaluators/LandIce_BasalMeltRate.cpp
  evaluators/LandIce_ColumnCouplingTestResidual.cpp
  evaluators/LandIce_Dissipation.cpp
  evaluators/LandIce_DOFDivInterpolationSide.cpp
  evaluators/LandIce_EnthalpyResid.cpp
  evaluators/LandIce_EnthalpyBasalResid.cpp
  evaluators/LandIce_FlowRate.cpp
  evaluators/LandIce_FluxDiv.cpp
  evaluators/LandIce_FluxDivergenceResidual.cpp
  evaluators/LandIce_Gather2DField.cpp
  evaluators/LandIce_GatherVerticallyContractedSolution.cpp
  evaluators/LandIce_IceOverburden.cpp
  evaluators/LandIce_LaplacianRegularizationResidual.cpp
  evaluators/LandIce_L2ProjectedBoundaryLaplacianResidual.cpp
  evaluators/LandIce_LiquidWaterFraction.cpp
  evaluators/LandIce_PressureCorrectedTemperature.cpp
  evaluators/LandIce_PressureMeltingEnthalpy.cpp
  evaluators/LandIce_ResponseUtilities.cpp
  evaluators/LandIce_ResponseSurfaceVelocityMismatch.cpp
  evaluators/LandIce_ResponseSMBMismatch.cpp
  evaluators/LandIce_ResponseGLFlux.cpp
  evaluators/LandIce_ResponseBoundarySquaredL2Norm.cpp
  evaluators/LandIce_ScatterResidual2D.cpp
  evaluators/LandIce_SimpleOperationEvaluator.cpp
  evaluators/LandIce_StokesFOBasalResid.cpp
  evaluators/LandIce_StokesFOLateralResid.cpp
  evaluators/LandIce_StokesFOResid.cpp
  evaluators/LandIce_StokesFOStress.cpp
  evaluators/LandIce_StokesFOSynteticTestBC.cpp
  evaluators/LandIce_StokesFOBodyForce.cpp
  evaluators/LandIce_StokesFOImplicitThicknessUpdateResid.cpp
  evaluators/LandIce_SurfaceAirEnthalpy.cpp
  evaluators/LandIce_Temperature.cpp
  evaluators/LandIce_ThicknessResid.cpp
  evaluators/LandIce_UpdateZCoordinate.cpp
  evaluators/LandIce_VerticalVelocity.cpp
  evaluators/LandIce_ViscosityFO.cpp
  evaluators/LandIce_w_Resid.cpp
  evaluators/LandIce_Time.cpp
  problems/LandIce_ColumnCouplingTest.cpp
  problems/LandIce_Enthalpy.cpp
  problems/LandIce_Hydrology.cpp
  problems/LandIce_LaplacianSampling.cpp
  problems/LandIce_ProblemFactory.cpp
  problems/LandIce_ProblemUtils.cpp
  problems/LandIce_StokesFO.cpp
  problems/LandIce_StokesFOBase.cpp
  problems/LandIce_StokesFOThermoCoupled.cpp
  problems/LandIce_StokesFOThickness.cpp
  problems/LandIce_StokesFOHydrology.cpp
  evaluators/hydrology/LandIce_HydrologyBasalGravitationalWaterPotential.cpp
  evaluators/hydrology/LandIce_EffectivePressure.cpp
  evaluators/hydrology/LandIce_HydraulicPotential.cpp
  evaluators/hydrology/LandIce_HydrologyMeltingRate.cpp
  evaluators/hydrology/LandIce_HydrologyResidualCavitiesEqn.cpp
  evaluators/hydrology/LandIce_HydrologyResidualMassEqn.cpp
  evaluators/hydrology/LandIce_HydrologyResidualTillStorageEqn.cpp
  evaluators/hydrology/LandIce_HydrologySurfaceWaterInput.cpp
  evaluators/hydrology/LandIce_HydrologyWaterDischarge.cpp
  evaluators/hydrology/LandIce_HydrologyWaterThickness.cpp
)

set(HEADERS
  evaluators/LandIce_BasalFrictionCoefficient.hpp
  evaluators/LandIce_BasalFrictionCoefficient_Def.hpp
  evaluators/LandIce_BasalMeltRate.hpp
  evaluators/LandIce_BasalMeltRate_Def.hpp
  evaluators/LandIce_ColumnCouplingTestResidual.hpp
  evaluators/LandIce_ColumnCouplingTestResidual_Def.hpp
  evaluators/LandIce_Dissipation.hpp
  evaluators/LandIce_Dissipation_Def.hpp
  evaluators/LandIce_DOFDivInterpolationSide.hpp
  evaluators/LandIce_DOFDivInterpolationSide_Def.hpp
  evaluators/LandIce_EnthalpyResid.hpp
  evaluators/LandIce_EnthalpyResid_Def.hpp
  evaluators/LandIce_EnthalpyBasalResid.hpp
  evaluators/LandIce_EnthalpyBasalResid_Def.hpp
  evaluators/LandIce_FlowRate.hpp
  evaluators/LandIce_FlowRate_Def.hpp
  evaluators/LandIce_FluxDiv.hpp
  evaluators/LandIce_FluxDiv_Def.hpp
  evaluators/LandIce_FluxDivergenceResidual.hpp
  evaluators/LandIce_FluxDivergenceResidual_Def.hpp
  evaluators/LandIce_Gather2DField.hpp
  evaluators/LandIce_Gather2DField_Def.hpp
  evaluators/LandIce_GatherVerticallyContractedSolution.hpp
  evaluators/LandIce_GatherVerticallyContractedSolution_Def.hpp
  evaluators/LandIce_IceOverburden.hpp
  evaluators/LandIce_IceOverburden_Def.hpp
  evaluators/LandIce_LaplacianRegularizationResidual.hpp
  evaluators/LandIce_LaplacianRegularizationResidual_Def.hpp
  evaluators/LandIce_L2ProjectedBoundaryLaplacianResidual.hpp
  evaluators/LandIce_L2ProjectedBoundaryLaplacianResidual_Def.hpp
  evaluators/LandIce_LiquidWaterFraction.hpp
  evaluators/LandIce_LiquidWaterFraction_Def.hpp
  evaluators/LandIce_PressureCorrectedTemperature.hpp
  evaluators/LandIce_PressureCorrectedTemperature_Def.hpp
  evaluators/LandIce_PressureMeltingEnthalpy.hpp
  evaluators/LandIce_PressureMeltingEnthalpy_Def.hpp
  evaluators/LandIce_ResponseUtilities.hpp
  evaluators/LandIce_ResponseUtilities_Def.hpp
  evaluators/LandIce_ResponseSurfaceVelocityMismatch.hpp
  evaluators/LandIce_ResponseSurfaceVelocityMismatch_Def.hpp
  evaluators/LandIce_ResponseBoundarySquaredL2Norm.hpp
  evaluators/LandIce_ResponseBoundarySquaredL2Norm_Def.hpp
  evaluators/LandIce_ResponseSMBMismatch.hpp
  evaluators/LandIce_ResponseSMBMismatch_Def.hpp
  evaluators/LandIce_ResponseGLFlux.hpp
  evaluators/LandIce_ResponseGLFlux_Def.hpp
  evaluators/LandIce_ScatterResidual2D.hpp
  evaluators/LandIce_ScatterResidual2D_Def.hpp
  evaluators/LandIce_SimpleOperation.hpp
  evaluators/LandIce_SimpleOperationEvaluator.hpp
  evaluators/LandIce_SimpleOperationEvaluator_Def.hpp
  evaluators/LandIce_StokesFOBodyForce.hpp
  evaluators/LandIce_StokesFOBodyForce_Def.hpp
  evaluators/LandIce_StokesFOLateralResid.hpp
  evaluators/LandIce_StokesFOLateralResid_Def.hpp
  evaluators/LandIce_StokesFOBasalResid.hpp
  evaluators/LandIce_StokesFOBasalResid_Def.hpp
  evaluators/LandIce_StokesFOImplicitThicknessUpdateResid.hpp
  evaluators/LandIce_StokesFOImplicitThicknessUpdateResid_Def.hpp
  evaluators/LandIce_StokesFOResid.hpp
  evaluators/LandIce_StokesFOResid_Def.hpp
  evaluators/LandIce_StokesFOStress.hpp
  evaluators/LandIce_StokesFOStress_Def.hpp
  evaluators/LandIce_StokesFOSynteticTestBC.hpp
  evaluators/LandIce_StokesFOSynteticTestBC_Def.hpp
  evaluators/LandIce_SurfaceAirEnthalpy.hpp
  evaluators/LandIce_SurfaceAirEnthalpy_Def.hpp
  evaluators/LandIce_Temperature.hpp
  evaluators/LandIce_Temperature_Def.hpp
  evaluators/LandIce_ThicknessResid_Def.hpp
  evaluators/LandIce_ThicknessResid.hpp
  evaluators/LandIce_UpdateZCoordinate.hpp
  evaluators/LandIce_UpdateZCoordinate_Def.hpp
  evaluators/LandIce_VerticalVelocity.hpp
  evaluators/LandIce_VerticalVelocity_Def.hpp
  evaluators/LandIce_ViscosityFO.hpp
  evaluators/LandIce_ViscosityFO_Def.hpp
  evaluators/LandIce_w_Resid.hpp
  evaluators/LandIce_w_Resid_Def.hpp
  evaluators/LandIce_Time.hpp
  evaluators/LandIce_Time_Def.hpp
  evaluators/hydrology/LandIce_HydrologyBasalGravitationalWaterPotential.hpp
  evaluators/hydrology/LandIce_HydrologyBasalGravitationalWaterPotential_Def.hpp
  evaluators/hydrology/LandIce_EffectivePressure.hpp
  evaluators/hydrology/LandIce_EffectivePressure_Def.hpp
  evaluators/hydrology/LandIce_HydraulicPotential.hpp
  evaluators/hydrology/LandIce_HydraulicPotential_Def.hpp
  evaluators/hydrology/LandIce_HydrologyMeltingRate.hpp
  evaluators/hydrology/LandIce_HydrologyMeltingRate_Def.hpp
  evaluators/hydrology/LandIce_HydrologyResidualCavitiesEqn.hpp
  evaluators/hydrology/LandIce_HydrologyResidualCavitiesEqn_Def.hpp
  evaluators/hydrology/LandIce_HydrologyResidualMassEqn.hpp
  evaluators/hydrology/LandIce_HydrologyResidualMassEqn_Def.hpp
  evaluators/hydrology/LandIce_HydrologyResidualTillStorageEqn.hpp
  evaluators/hydrology/LandIce_HydrologyResidualTillStorageEqn_Def.hpp
  evaluators/hydrology/LandIce_HydrologySurfaceWaterInput.hpp
  evaluators/hydrology/LandIce_HydrologySurfaceWaterInput_Def.hpp
  evaluators/hydrology/LandIce_HydrologyWaterDischarge.hpp
  evaluators/hydrology/LandIce_HydrologyWaterDischarge_Def.hpp
  evaluators/hydrology/LandIce_HydrologyWaterThickness.hpp
  evaluators/hydrology/LandIce_HydrologyWaterThickness_Def.hpp
  utility/Intrepid2_HGRAD_LINE_I4_FEM.hpp
  utility/Intrepid2_HGRAD_LINE_I4_FEMDef.hpp
  problems/LandIce_ColumnCouplingTest.hpp
  problems/LandIce_Enthalpy.hpp
  problems/LandIce_Hydrology.hpp
  problems/LandIce_LaplacianSampling.hpp
  problems/LandIce_ParamEnum.hpp
  problems/LandIce_ProblemFactory.hpp
  problems/LandIce_StokesFOBase.hpp
  problems/LandIce_StokesFO.hpp
  problems/LandIce_StokesFOThermoCoupled.hpp
  problems/LandIce_StokesFOThickness.hpp
  problems/LandIce_StokesFOHydrology.hpp
)

# This is not very satisfactory, since it injects CISM specific stuff into
# the main landIce library. But StokesFO needs the definition of these
# when building for CISM, or we run into a link error.
if(ENABLE_CISM_INTERFACE)
  list (APPEND SOURCES evaluators/LandIce_CismSurfaceGradFO.cpp)
  list (APPEND HEADERS
    evaluators/LandIce_CismSurfaceGradFO.hpp
    evaluators/LandIce_CismSurfaceGradFO_Def.hpp)
endif()

add_library(landIce ${Albany_LIBRARY_TYPE} ${SOURCES})
set_target_properties(landIce PROPERTIES PUBLIC_HEADER "${HEADERS}")
target_link_libraries(landIce PUBLIC albanyLib)
if (ALBANY_SUPPRESS_TRILINOS_WARNINGS)
  target_include_directories(landIce SYSTEM PUBLIC
                            "${Trilinos_INCLUDE_DIRS};${Trilinos_TPL_INCLUDE_DIRS}")
else()
  target_include_directories(landIce PUBLIC
                            "${Trilinos_INCLUDE_DIRS};${Trilinos_TPL_INCLUDE_DIRS}")
endif()
target_compile_definitions (landIce PUBLIC  ALBANY_LANDICE)
if (ENABLE_CISM_INTERFACE)
  target_compile_definitions (landIce PUBLIC CISM_HAS_LANDICE)
endif()

set (LANDICE_INCLUDE_DIRS
     ${ALBANY_INCLUDE_DIRS}
     ${CMAKE_CURRENT_SOURCE_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}/evaluators
     ${CMAKE_CURRENT_SOURCE_DIR}/evaluators/hydrology
     ${CMAKE_CURRENT_SOURCE_DIR}/utility
     ${CMAKE_CURRENT_SOURCE_DIR}/problems
)

target_include_directories(landIce PUBLIC
                           "$<BUILD_INTERFACE:${LANDICE_INCLUDE_DIRS}>"
                            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

set (ALBANY_LIBRARIES ${ALBANY_LIBRARIES} landIce)

install(TARGETS landIce EXPORT albany-export
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/"
        PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

###################################
###     CISM interface lib     ####
###################################

if(ENABLE_CISM_INTERFACE)
  add_subdirectory(interfaceWithCISM)
  set (ALBANY_LIBRARIES ${ALBANY_LIBRARIES} cismInterface)
endif()

###################################
###     MPAS interface lib     ####
###################################

if(ENABLE_MPAS_INTERFACE)
  add_subdirectory(interfaceWithMPAS)
  set (ALBANY_LIBRARIES ${ALBANY_LIBRARIES} mpasInterface)
endif()

set (ALBANY_LIBRARIES ${ALBANY_LIBRARIES} PARENT_SCOPE)
