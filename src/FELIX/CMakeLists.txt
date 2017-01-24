##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//
# FELIX


  SET(SOURCES
       evaluators/FELIX_BasalFrictionCoefficient.cpp
       evaluators/FELIX_BasalFrictionCoefficientGradient.cpp
       evaluators/FELIX_BasalFrictionCoefficientNode.cpp
       evaluators/FELIX_BasalFrictionHeat.cpp
       evaluators/FELIX_BasalMeltRate.cpp
       evaluators/FELIX_DirichletEnthalpySurface.cpp
       evaluators/FELIX_Dissipation.cpp
       evaluators/FELIX_DummyResidual.cpp
       evaluators/FELIX_EffectivePressure.cpp
       evaluators/FELIX_EnthalpyResid.cpp
       evaluators/FELIX_EnthalpyBasalResid.cpp
       evaluators/FELIX_EpsilonL1L2.cpp
       evaluators/FELIX_FlowRate.cpp
       evaluators/FELIX_FluxDiv.cpp
       evaluators/FELIX_GeoFluxHeat.cpp
       evaluators/FELIX_HydrologyDirichlet.cpp
       evaluators/FELIX_HydrologyWaterDischarge.cpp
       evaluators/FELIX_HydrologyWaterSource.cpp
       evaluators/FELIX_HydrologyMeltingRate.cpp
       evaluators/FELIX_HydrologyResidualPotentialEqn.cpp
       evaluators/FELIX_HydrologyResidualThicknessEqn.cpp
       evaluators/FELIX_HydrostaticPressure.cpp
       evaluators/FELIX_Integral1Dw_Z.cpp
       evaluators/FELIX_LiquidWaterFraction.cpp
       evaluators/FELIX_PressureMeltingTemperature.cpp
       evaluators/FELIX_PressureMeltingEnthalpy.cpp
       evaluators/FELIX_ProlongateVector.cpp
       evaluators/FELIX_L2ProjectedBoundaryLaplacianResidual.cpp
       evaluators/FELIX_LaplacianRegularizationResidual.cpp
       evaluators/FELIX_ResponseSurfaceVelocityMismatch.cpp
       evaluators/FELIX_ResponseSMBMismatch.cpp
       evaluators/FELIX_ResponseBoundarySquaredL2Norm.cpp
       evaluators/FELIX_SimpleOperation.cpp
       evaluators/FELIX_StackFields.cpp
       evaluators/FELIX_StokesMomentumResid.cpp
       evaluators/FELIX_StokesFOBasalResid.cpp
       evaluators/FELIX_StokesFOResid.cpp
       evaluators/FELIX_StokesFOStress.cpp
       evaluators/FELIX_StokesContinuityResid.cpp
       evaluators/FELIX_StokesContravarientMetricTensor.cpp
       evaluators/FELIX_StokesBodyForce.cpp
       evaluators/FELIX_StokesFOBodyForce.cpp
       evaluators/FELIX_StokesL1L2BodyForce.cpp
       evaluators/FELIX_StokesL1L2Resid.cpp
       evaluators/FELIX_StokesRm.cpp
       evaluators/FELIX_StokesTauM.cpp
       evaluators/FELIX_Temperature.cpp
       evaluators/FELIX_ThicknessResid.cpp
       evaluators/FELIX_VerticalVelocity.cpp
       evaluators/FELIX_Viscosity.cpp
       evaluators/FELIX_ViscosityFO.cpp
       evaluators/FELIX_ViscosityL1L2.cpp
       evaluators/FELIX_w_ZResid.cpp
       evaluators/FELIX_w_Resid.cpp
       evaluators/FELIX_Time.cpp
       problems/FELIX_ProblemFactory.cpp
       problems/FELIX_Enthalpy.cpp
       problems/FELIX_Hydrology.cpp
       problems/FELIX_PopulateMesh.cpp
       problems/FELIX_SchoofFit.cpp
       problems/FELIX_Stokes.cpp
       problems/FELIX_StokesFO.cpp
       problems/FELIX_LaplacianSampling.cpp
       problems/FELIX_StokesFOThermoCoupled.cpp
       problems/FELIX_StokesL1L2.cpp
   )

IF (ALBANY_EPETRA)
  SET(SOURCES ${SOURCES}
       evaluators/FELIX_StokesFOImplicitThicknessUpdateResid.cpp
       evaluators/FELIX_Gather2DField.cpp
       evaluators/FELIX_UpdateZCoordinate.cpp
       evaluators/FELIX_GatherVerticallyAveragedVelocity.cpp
       evaluators/FELIX_ScatterResidual2D.cpp
       problems/FELIX_StokesFOThickness.cpp
       problems/FELIX_StokesFOHydrology.cpp
       problems/FELIX_Hydrology.cpp
   )
ENDIF()

IF(ENABLE_MPAS_INTERFACE)
   SET(SOURCESMPAS
       interface_with_mpas/Albany_MpasSTKMeshStruct.cpp
       interface_with_mpas/Interface.cpp
   )
ENDIF()

IF(ENABLE_CISM_INTERFACE)
   SET(SOURCESCISM
       interface_with_cism/Albany_CismSTKMeshStruct.cpp
       interface_with_cism/felix_driver.cpp
       evaluators/FELIX_CismSurfaceGradFO.cpp
   )
ENDIF()

  SET(HEADERS
       evaluators/FELIX_BasalFrictionCoefficient.hpp
       evaluators/FELIX_BasalFrictionCoefficient_Def.hpp
       evaluators/FELIX_BasalFrictionCoefficientGradient.hpp
       evaluators/FELIX_BasalFrictionCoefficientGradient_Def.hpp
       evaluators/FELIX_BasalFrictionCoefficientNode.hpp
       evaluators/FELIX_BasalFrictionCoefficientNode_Def.hpp
       evaluators/FELIX_BasalFrictionHeat.hpp
       evaluators/FELIX_BasalFrictionHeat_Def.hpp
       evaluators/FELIX_BasalMeltRate.hpp
       evaluators/FELIX_BasalMeltRate_Def.hpp
       evaluators/FELIX_DirichletEnthalpySurface.hpp
       evaluators/FELIX_DirichletEnthalpySurface_Def.hpp
       evaluators/FELIX_Dissipation.hpp
       evaluators/FELIX_Dissipation_Def.hpp
       evaluators/FELIX_DummyResidual.hpp
       evaluators/FELIX_DummyResidual_Def.hpp
       evaluators/FELIX_EffectivePressure.hpp
       evaluators/FELIX_EffectivePressure_Def.hpp
       evaluators/FELIX_EnthalpyResid.hpp
       evaluators/FELIX_EnthalpyResid_Def.hpp
       evaluators/FELIX_EnthalpyBasalResid.hpp
       evaluators/FELIX_EnthalpyBasalResid_Def.hpp
       evaluators/FELIX_EpsilonL1L2.hpp
       evaluators/FELIX_EpsilonL1L2_Def.hpp
       evaluators/FELIX_FlowRate.hpp
       evaluators/FELIX_FlowRate_Def.hpp
       evaluators/FELIX_FluxDiv.hpp
       evaluators/FELIX_FluxDiv_Def.hpp
       evaluators/FELIX_GeoFluxHeat.hpp
       evaluators/FELIX_GeoFluxHeat_Def.hpp
       evaluators/FELIX_HydrologyDirichlet.hpp
       evaluators/FELIX_HydrologyDirichlet_Def.hpp
       evaluators/FELIX_HydrologyWaterDischarge.hpp
       evaluators/FELIX_HydrologyWaterDischarge_Def.hpp
       evaluators/FELIX_HydrologyWaterSource.hpp
       evaluators/FELIX_HydrologyWaterSource_Def.hpp
       evaluators/FELIX_HydrologyMeltingRate.hpp
       evaluators/FELIX_HydrologyMeltingRate_Def.hpp
       evaluators/FELIX_HydrologyResidualPotentialEqn.hpp
       evaluators/FELIX_HydrologyResidualPotentialEqn_Def.hpp
       evaluators/FELIX_HydrologyResidualThicknessEqn.hpp
       evaluators/FELIX_HydrologyResidualThicknessEqn_Def.hpp
       evaluators/FELIX_HydrostaticPressure.hpp
       evaluators/FELIX_HydrostaticPressure_Def.hpp
       evaluators/FELIX_Integral1Dw_Z.hpp
       evaluators/FELIX_Integral1Dw_Z_Def.hpp
       evaluators/FELIX_LiquidWaterFraction.hpp
       evaluators/FELIX_LiquidWaterFraction_Def.hpp
       evaluators/FELIX_PressureMeltingTemperature.hpp
       evaluators/FELIX_PressureMeltingTemperature_Def.hpp
       evaluators/FELIX_PressureMeltingEnthalpy.hpp
       evaluators/FELIX_PressureMeltingEnthalpy_Def.hpp
       evaluators/FELIX_ProlongateVector.hpp
       evaluators/FELIX_ProlongateVector_Def.hpp
       evaluators/FELIX_L2ProjectedBoundaryLaplacianResidual.hpp
       evaluators/FELIX_L2ProjectedBoundaryLaplacianResidual_Def.hpp
       evaluators/FELIX_LaplacianRegularizationResidual.hpp
       evaluators/FELIX_LaplacianRegularizationResidual_Def.hpp
       evaluators/FELIX_ResponseSurfaceVelocityMismatch.hpp
       evaluators/FELIX_ResponseSurfaceVelocityMismatch_Def.hpp
       evaluators/FELIX_ResponseBoundarySquaredL2Norm.hpp
       evaluators/FELIX_ResponseBoundarySquaredL2Norm_Def.hpp
       evaluators/FELIX_ResponseSMBMismatch.hpp
       evaluators/FELIX_ResponseSMBMismatch_Def.hpp
       evaluators/FELIX_SharedParameter.hpp
       evaluators/FELIX_SimpleOperation.hpp
       evaluators/FELIX_SimpleOperation_Def.hpp
       evaluators/FELIX_StackFields.hpp
       evaluators/FELIX_StackFields_Def.hpp
       evaluators/FELIX_StokesBodyForce.hpp
       evaluators/FELIX_StokesBodyForce_Def.hpp
       evaluators/FELIX_StokesContinuityResid.hpp
       evaluators/FELIX_StokesContinuityResid_Def.hpp
       evaluators/FELIX_StokesContravarientMetricTensor.hpp
       evaluators/FELIX_StokesContravarientMetricTensor_Def.hpp
       evaluators/FELIX_StokesFOBodyForce.hpp
       evaluators/FELIX_StokesFOBodyForce_Def.hpp
       evaluators/FELIX_StokesFOBasalResid.hpp
       evaluators/FELIX_StokesFOBasalResid_Def.hpp
       evaluators/FELIX_StokesFOResid.hpp
       evaluators/FELIX_StokesFOResid_Def.hpp
       evaluators/FELIX_StokesFOStress.hpp
       evaluators/FELIX_StokesFOStress_Def.hpp
       evaluators/FELIX_StokesL1L2Resid_Def.hpp
       evaluators/FELIX_StokesL1L2Resid.hpp
       evaluators/FELIX_StokesMomentumResid.hpp
       evaluators/FELIX_StokesMomentumResid_Def.hpp
       evaluators/FELIX_StokesRm.hpp
       evaluators/FELIX_StokesRm_Def.hpp
       evaluators/FELIX_StokesTauM.hpp
       evaluators/FELIX_StokesTauM_Def.hpp
       evaluators/FELIX_Temperature.hpp
       evaluators/FELIX_Temperature_Def.hpp
       evaluators/FELIX_ThicknessResid_Def.hpp
       evaluators/FELIX_ThicknessResid.hpp
       evaluators/FELIX_VerticalVelocity.hpp
       evaluators/FELIX_VerticalVelocity_Def.hpp
       evaluators/FELIX_ViscosityFO.hpp
       evaluators/FELIX_ViscosityFO_Def.hpp
       evaluators/FELIX_ViscosityL1L2.hpp
       evaluators/FELIX_ViscosityL1L2_Def.hpp
       evaluators/FELIX_w_ZResid.hpp
       evaluators/FELIX_w_ZResid_Def.hpp
       evaluators/FELIX_w_Resid.hpp
       evaluators/FELIX_w_Resid_Def.hpp
       evaluators/FELIX_Time.hpp
       evaluators/FELIX_Time_Def.hpp
       problems/FELIX_ProblemFactory.hpp
       problems/FELIX_Enthalpy.hpp
       problems/FELIX_StokesFOThermoCoupled.hpp
       problems/FELIX_Hydrology.hpp
       problems/FELIX_ParamEnum.hpp
       problems/FELIX_PopulateMesh.hpp
       problems/FELIX_SchoofFit.hpp
       problems/FELIX_Stokes.hpp
       problems/FELIX_StokesFO.hpp
       problems/FELIX_LaplacianSampling.hpp
       problems/FELIX_StokesL1L2.hpp
  )

IF (ALBANY_EPETRA)
  SET(HEADERS ${HEADERS}
       evaluators/FELIX_Gather2DField.hpp
       evaluators/FELIX_Gather2DField_Def.hpp
       evaluators/FELIX_GatherVerticallyAveragedVelocity.hpp
       evaluators/FELIX_GatherVerticallyAveragedVelocity_Def.hpp
       evaluators/FELIX_ScatterResidual2D.hpp
       evaluators/FELIX_ScatterResidual2D_Def.hpp
       evaluators/FELIX_UpdateZCoordinate.hpp
       evaluators/FELIX_UpdateZCoordinate_Def.hpp
       evaluators/FELIX_StokesFOImplicitThicknessUpdateResid.hpp
       evaluators/FELIX_StokesFOImplicitThicknessUpdateResid_Def.hpp
       problems/FELIX_StokesFOThickness.hpp
       problems/FELIX_StokesFOHydrology.hpp
       problems/FELIX_Hydrology.hpp
  )
ENDIF()

IF(ENABLE_MPAS_INTERFACE)
  SET(HEADERSMPAS
       interface_with_mpas/Interface.hpp
  )
ENDIF()

IF (ENABLE_CISM_INTERFACE)
  SET(HEADERSCISM
       interface_with_cism/felix_driver.H
       evaluators/FELIX_CismSurfaceGradFO_Def.hpp
       evaluators/FELIX_CismSurfaceGradFO.hpp
   )
ENDIF()

IF(ENABLE_CISM_INTERFACE)
  IF (ENABLE_CISM_EPETRA)
    add_library(cismInterfaceE ${Albany_LIBRARY_TYPE} ${SOURCESCISM}
                ${CISM_INCLUDE_DIR}/DyCoreToGlimmer.cpp
                ${HEADERSCISM} ${CISM_INCLUDE_DIR}/DyCoreToGlimmer.H)
    set_target_properties(cismInterfaceE PROPERTIES PUBLIC_HEADER "${HEADERSCISM}")
    include_directories (${Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
                         ${CISM_INCLUDE_DIR} ${CISM_INCLUDE_DIR}/FELIX
                         ${Albany_SOURCE_DIR}/src ${Albany_SOURCE_DIR}/src/evaluators
                         ${Albany_SOURCE_DIR}/src/problems
                         ${Albany_SOURCE_DIR}/src/FELIX/interface_with_cism ${Albany_SOURCE_DIR}/src/responses
                         ./problems ./evaluators
                        )
    IF (INSTALL_ALBANY)
      install(TARGETS cismInterfaceE EXPORT albany-export
        LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
        ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
        PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
    ENDIF()
  ELSE()
    add_library(cismInterface ${Albany_LIBRARY_TYPE} ${SOURCESCISM}
                ${CISM_INCLUDE_DIR}/DyCoreToGlimmer.cpp
                ${HEADERSCISM} ${CISM_INCLUDE_DIR}/DyCoreToGlimmer.H)
    set_target_properties(cismInterface PROPERTIES PUBLIC_HEADER "${HEADERSCISM}")
    include_directories (${Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
                         ${CISM_INCLUDE_DIR} ${CISM_INCLUDE_DIR}/FELIX
                         ${Albany_SOURCE_DIR}/src ${Albany_SOURCE_DIR}/src/evaluators
                         ${Albany_SOURCE_DIR}/src/problems
                         ${Albany_SOURCE_DIR}/src/FELIX/interface_with_cism ${Albany_SOURCE_DIR}/src/responses
                         ./problems ./evaluators
                        )
    IF (INSTALL_ALBANY)
      install(TARGETS cismInterface EXPORT albany-export
        LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
        ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
        PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
    ENDIF()
  ENDIF()
ELSE()
  include_directories (${Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
                       ${Albany_SOURCE_DIR}/src ${Albany_SOURCE_DIR}/src/evaluators
                       ${Albany_SOURCE_DIR}/src/problems ${Albany_SOURCE_DIR}/src/responses
                       ./problems ./evaluators
                      )
ENDIF()

add_library(FELIX ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})
set_target_properties(FELIX PROPERTIES PUBLIC_HEADER "${HEADERS}")

IF(ENABLE_MPAS_INTERFACE)
  add_library(mpasInterface STATIC ${SOURCESMPAS} ${HEADERSMPAS})
  set_target_properties(mpasInterface PROPERTIES PUBLIC_HEADER "${HEADERSMPAS}")
  IF (INSTALL_ALBANY)
    install(TARGETS mpasInterface
      LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
      ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
      PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
  ENDIF()
ENDIF()

IF (INSTALL_ALBANY)
  install(TARGETS FELIX EXPORT albany-export
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()


