##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//
# FELIX


  SET(SOURCES
       evaluators/FELIX_BasalFrictionCoefficient.cpp
       evaluators/FELIX_EffectivePressure.cpp
       evaluators/FELIX_EpsilonL1L2.cpp
       evaluators/FELIX_FieldNorm.cpp
       evaluators/FELIX_HydrologyDischarge.cpp
       evaluators/FELIX_HydrologyHydrostaticPotential.cpp
       evaluators/FELIX_HydrologyMelting.cpp
       evaluators/FELIX_HydrologyResidual.cpp
       evaluators/FELIX_HydrologyRhs.cpp
       evaluators/FELIX_QuadPointsToCellInterpolation.cpp
       evaluators/FELIX_StokesMomentumResid.cpp
       evaluators/FELIX_StokesFOResid.cpp
       evaluators/FELIX_ThicknessResid.cpp
       evaluators/FELIX_StokesL1L2Resid.cpp
       evaluators/FELIX_Viscosity.cpp
       evaluators/FELIX_ViscosityFO.cpp
       evaluators/FELIX_ViscosityL1L2.cpp
       evaluators/FELIX_StokesContinuityResid.cpp
       evaluators/FELIX_StokesTauM.cpp
       evaluators/FELIX_StokesContravarientMetricTensor.cpp
       evaluators/FELIX_StokesRm.cpp
       evaluators/FELIX_StokesBodyForce.cpp
       evaluators/FELIX_StokesFOBodyForce.cpp
       evaluators/FELIX_StokesL1L2BodyForce.cpp
       problems/FELIX_Hydrology.cpp
       evaluators/FELIX_ResponseSurfaceVelocityMismatch.cpp
       problems/FELIX_Stokes.cpp
       problems/FELIX_StokesFO.cpp
       problems/FELIX_StokesL1L2.cpp
   )

IF (ALBANY_EPETRA)
  SET(SOURCES ${SOURCES}
       evaluators/FELIX_StokesFOImplicitThicknessUpdateResid.cpp
       evaluators/FELIX_GatherThickness.cpp
       evaluators/FELIX_GatherVerticallyAveragedVelocity.cpp
       evaluators/FELIX_ScatterResidualH.cpp

       problems/FELIX_StokesFOThickness.cpp
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
       evaluators/FELIX_EffectivePressure.hpp
       evaluators/FELIX_EffectivePressure_Def.hpp
       evaluators/FELIX_EpsilonL1L2.hpp
       evaluators/FELIX_EpsilonL1L2_Def.hpp
       evaluators/FELIX_FieldNorm.hpp
       evaluators/FELIX_FieldNorm_Def.hpp
       evaluators/FELIX_HydrologyDischarge.hpp
       evaluators/FELIX_HydrologyDischarge_Def.hpp
       evaluators/FELIX_HydrologyHydrostaticPotential.hpp
       evaluators/FELIX_HydrologyHydrostaticPotential_Def.hpp
       evaluators/FELIX_HydrologyMelting.hpp
       evaluators/FELIX_HydrologyMelting_Def.hpp
       evaluators/FELIX_HydrologyResidual.hpp
       evaluators/FELIX_HydrologyResidual_Def.hpp
       evaluators/FELIX_HydrologyRhs.hpp
       evaluators/FELIX_HydrologyRhs_Def.hpp
       evaluators/FELIX_QuadPointsToCellInterpolation.hpp
       evaluators/FELIX_QuadPointsToCellInterpolation_Def.hpp
       evaluators/FELIX_StokesMomentumResid.hpp
       evaluators/FELIX_StokesFOResid_Def.hpp
       evaluators/FELIX_StokesFOResid.hpp
       evaluators/FELIX_ThicknessResid_Def.hpp
       evaluators/FELIX_ThicknessResid.hpp
       evaluators/FELIX_StokesL1L2Resid_Def.hpp
       evaluators/FELIX_StokesL1L2Resid.hpp
       evaluators/FELIX_StokesMomentumResid_Def.hpp
       evaluators/FELIX_ViscosityFO.hpp
       evaluators/FELIX_ViscosityFO_Def.hpp
       evaluators/FELIX_ViscosityL1L2.hpp
       evaluators/FELIX_ViscosityL1L2_Def.hpp
       evaluators/FELIX_StokesContinuityResid.hpp
       evaluators/FELIX_StokesContinuityResid_Def.hpp
       evaluators/FELIX_StokesTauM.hpp
       evaluators/FELIX_StokesTauM_Def.hpp
       evaluators/FELIX_StokesContravarientMetricTensor.hpp
       evaluators/FELIX_StokesContravarientMetricTensor_Def.hpp
       evaluators/FELIX_StokesRm.hpp
       evaluators/FELIX_StokesRm_Def.hpp
       evaluators/FELIX_StokesBodyForce.hpp
       evaluators/FELIX_StokesBodyForce_Def.hpp
       evaluators/FELIX_StokesFOBodyForce.hpp
       evaluators/FELIX_StokesFOBodyForce_Def.hpp
       problems/FELIX_Hydrology.hpp
       evaluators/FELIX_ResponseSurfaceVelocityMismatch.hpp
       evaluators/FELIX_ResponseSurfaceVelocityMismatch_Def.hpp
       problems/FELIX_Stokes.hpp
       problems/FELIX_StokesFO.hpp
       problems/FELIX_StokesL1L2.hpp
  )

IF (ALBANY_EPETRA)
  SET(HEADERS ${HEADERS}
       evaluators/FELIX_StokesFOImplicitThicknessUpdateResid.hpp
       evaluators/FELIX_StokesFOImplicitThicknessUpdateResid_Def.hpp
       evaluators/FELIX_GatherThickness.hpp
       evaluators/FELIX_GatherThickness_Def.hpp
       evaluators/FELIX_GatherVerticallyAveragedVelocity.hpp
       evaluators/FELIX_GatherVerticallyAveragedVelocity_Def.hpp
       evaluators/FELIX_ScatterResidualH.hpp
       evaluators/FELIX_ScatterResidualH_Def.hpp

       problems/FELIX_StokesFOThickness.hpp
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


