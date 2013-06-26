##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# LCM problems
SET(SOURCES 
  problems/ElasticityProblem.cpp
  problems/MechanicsProblem.cpp
  problems/ThermoElasticityProblem.cpp
  problems/GradientDamageProblem.cpp
  problems/PoroElasticityProblem.cpp
  problems/UnSatPoroElasticityProblem.cpp
  problems/TLPoroPlasticityProblem.cpp
  problems/ThermoPoroPlasticityProblem.cpp
  problems/ThermoMechanicalProblem.cpp
  problems/HDiffusionDeformationProblem.cpp
  problems/ProjectionProblem.cpp
  problems/FieldNameMap.cpp
  )

SET(HEADERS 
  problems/ElasticityProblem.hpp
  problems/MechanicsProblem.hpp
  problems/ThermoElasticityProblem.hpp
  problems/GradientDamageProblem.hpp
  problems/PoroElasticityProblem.hpp
  problems/UnSatPoroElasticityProblem.hpp
  problems/TLPoroPlasticityProblem.hpp
  problems/ThermoPoroPlasticityProblem.hpp
  problems/ThermoMechanicalProblem.hpp
  problems/HDiffusionDeformationProblem.hpp
  problems/ProjectionProblem.hpp
  problems/FieldNameMap.hpp
  )

# LCM evaluators
SET(SOURCES ${SOURCES}
  evaluators/Stress.cpp
  evaluators/CapExplicit.cpp
  evaluators/CapImplicit.cpp
  evaluators/Strain.cpp
  evaluators/ElasticModulus.cpp
  evaluators/PoissonsRatio.cpp
  evaluators/DefGrad.cpp
  evaluators/Neohookean.cpp
  evaluators/KirchhoffStVenant.cpp
  evaluators/J2Stress.cpp
  evaluators/ElasticityResid.cpp
  evaluators/TLElasResid.cpp
  evaluators/MechanicsResidual.cpp
  evaluators/HardeningModulus.cpp
  evaluators/YieldStrength.cpp
  evaluators/PisdWdF.cpp
  evaluators/DamageResid.cpp
  evaluators/J2Damage.cpp
  evaluators/DamageLS.cpp
  evaluators/SaturationModulus.cpp
  evaluators/SaturationExponent.cpp
  evaluators/SurfaceBasis.cpp
  evaluators/SurfaceScalarJump.cpp
  evaluators/SurfaceScalarGradient.cpp
  evaluators/SurfaceScalarGradientOperator.cpp
  evaluators/SurfaceDiffusionResidual.cpp
  evaluators/SurfaceTLPoroMassResidual.cpp
  evaluators/SurfaceVectorJump.cpp
  evaluators/SurfaceVectorGradient.cpp
  evaluators/SurfaceVectorResidual.cpp
  evaluators/CurrentCoords.cpp
  evaluators/DamageSource.cpp
  evaluators/BulkModulus.cpp
  evaluators/ShearModulus.cpp
  evaluators/KfieldBC.cpp
  evaluators/TorsionBC.cpp
  evaluators/DislocationDensity.cpp
  evaluators/TotalStress.cpp
  evaluators/BiotCoefficient.cpp
  evaluators/BiotModulus.cpp
  evaluators/PoroElasticityResidMomentum.cpp
  evaluators/PoroElasticityResidMass.cpp
  evaluators/UnSatPoroElasticityResidMass.cpp
  evaluators/TLPoroPlasticityResidMomentum.cpp
  evaluators/TLPoroPlasticityResidMass.cpp
  evaluators/TLPoroStress.cpp
  evaluators/ThermoPoroPlasticityResidMomentum.cpp
  evaluators/ThermoPoroPlasticityResidMass.cpp
  evaluators/ThermoPoroPlasticityResidEnergy.cpp
  evaluators/Porosity.cpp
  evaluators/KCPermeability.cpp
  evaluators/VanGenuchtenPermeability.cpp
  evaluators/VanGenuchtenSaturation.cpp
  evaluators/ThermoMechanicalEnergyResidual.cpp
  evaluators/ThermoMechanicalMomentumResidual.cpp
  evaluators/ThermoMechanicalStress.cpp
  evaluators/Time.cpp
  evaluators/TimeDepBC.cpp
  evaluators/TimeTracBC.cpp
  evaluators/DiffusionCoefficient.cpp
  evaluators/EffectiveDiffusivity.cpp
  evaluators/EquilibriumConstant.cpp
  evaluators/TrappedSolvent.cpp
  evaluators/TrappedConcentration.cpp
  evaluators/TransportCoefficients.cpp
  evaluators/TotalConcentration.cpp
  evaluators/HDiffusionDeformationMatterResidual.cpp
  evaluators/ScalarL2ProjectionResidual.cpp
  evaluators/L2ProjectionResidual.cpp
  evaluators/TauContribution.cpp
  evaluators/StrainRateFactor.cpp
  evaluators/MixtureThermalExpansion.cpp
  evaluators/MixtureSpecificHeat.cpp
  evaluators/StabParameter.cpp
  evaluators/GradientElementLength.cpp
  evaluators/UnitGradient.cpp
  evaluators/SetField.cpp
  evaluators/AssumedStrain.cpp
  evaluators/MooneyRivlinDamage.cpp
  evaluators/MooneyRivlin_Incompressible.cpp
  evaluators/MooneyRivlin_Incompressible_Damage.cpp
  evaluators/RecoveryModulus.cpp
  evaluators/FaceFractureCriteria.cpp
  evaluators/FaceAverage.cpp
  evaluators/AAA.cpp
  evaluators/LatticeDefGrad.cpp
  evaluators/TvergaardHutchinson.cpp
  evaluators/SurfaceCohesiveResidual.cpp
  evaluators/ConstitutiveModelInterface.cpp
  evaluators/ConstitutiveModelParameters.cpp
  evaluators/Kinematics.cpp
  evaluators/BifurcationCheck.cpp
  evaluators/TransportResidual.cpp
  evaluators/ThermoMechanicalCoefficients.cpp
  )

SET(HEADERS ${HEADERS}
  evaluators/Stress.hpp
  evaluators/Stress_Def.hpp
  evaluators/CapExplicit.hpp
  evaluators/CapExplicit_Def.hpp
  evaluators/CapImplicit.hpp
  evaluators/CapImplicit_Def.hpp
  evaluators/Strain.hpp
  evaluators/Strain_Def.hpp
  evaluators/ElasticModulus.hpp
  evaluators/ElasticModulus_Def.hpp
  evaluators/PoissonsRatio.hpp
  evaluators/PoissonsRatio_Def.hpp
  evaluators/DefGrad.hpp
  evaluators/DefGrad_Def.hpp
  evaluators/Neohookean.hpp
  evaluators/Neohookean_Def.hpp
  evaluators/KirchhoffStVenant.hpp
  evaluators/KirchhoffStVenant_Def.hpp
  evaluators/J2Stress.hpp
  evaluators/J2Stress_Def.hpp
  evaluators/ElasticityResid.hpp
  evaluators/ElasticityResid_Def.hpp
  evaluators/TLElasResid.hpp
  evaluators/TLElasResid_Def.hpp
  evaluators/MechanicsResidual.hpp
  evaluators/MechanicsResidual_Def.hpp
  evaluators/HardeningModulus.hpp
  evaluators/HardeningModulus_Def.hpp
  evaluators/YieldStrength.hpp
  evaluators/YieldStrength_Def.hpp
  evaluators/PisdWdF.hpp
  evaluators/PisdWdF_Def.hpp
  evaluators/DamageResid.hpp
  evaluators/DamageResid_Def.hpp
  evaluators/J2Damage.hpp
  evaluators/J2Damage_Def.hpp
  evaluators/DamageLS.hpp
  evaluators/DamageLS_Def.hpp
  evaluators/SaturationModulus.hpp
  evaluators/SaturationModulus_Def.hpp
  evaluators/SaturationExponent.hpp
  evaluators/SaturationExponent_Def.hpp
  evaluators/SurfaceBasis.hpp
  evaluators/SurfaceBasis_Def.hpp
  evaluators/SurfaceScalarJump.hpp
  evaluators/SurfaceScalarJump_Def.hpp
  evaluators/SurfaceVectorJump.hpp
  evaluators/SurfaceVectorJump_Def.hpp
  evaluators/SurfaceVectorGradient.hpp
  evaluators/SurfaceVectorGradient_Def.hpp
  evaluators/SurfaceScalarGradient.hpp
  evaluators/SurfaceScalarGradient_Def.hpp
  evaluators/SurfaceScalarGradientOperator.hpp
  evaluators/SurfaceScalarGradientOperator_Def.hpp
  evaluators/SurfaceDiffusionResidual.hpp
  evaluators/SurfaceDiffusionResidual_Def.hpp
  evaluators/SurfaceTLPoroMassResidual.hpp
  evaluators/SurfaceTLPoroMassResidual_Def.hpp
  evaluators/SurfaceVectorResidual.hpp
  evaluators/SurfaceVectorResidual_Def.hpp
  evaluators/CurrentCoords.hpp
  evaluators/CurrentCoords_Def.hpp
  evaluators/DamageSource.hpp
  evaluators/DamageSource_Def.hpp
  evaluators/BulkModulus.hpp
  evaluators/BulkModulus_Def.hpp
  evaluators/ShearModulus.hpp
  evaluators/ShearModulus_Def.hpp
  evaluators/KfieldBC.hpp
  evaluators/KfieldBC_Def.hpp
  evaluators/TorsionBC.hpp
  evaluators/TorsionBC_Def.hpp
  evaluators/DislocationDensity.hpp
  evaluators/DislocationDensity_Def.hpp
  evaluators/TotalStress.hpp
  evaluators/TotalStress_Def.hpp
  evaluators/TLPoroStress.hpp
  evaluators/TLPoroStress_Def.hpp
  evaluators/BiotCoefficient.hpp
  evaluators/BiotCoefficient_Def.hpp
  evaluators/BiotModulus.hpp
  evaluators/BiotModulus_Def.hpp
  evaluators/PoroElasticityResidMomentum.hpp
  evaluators/PoroElasticityResidMomentum_Def.hpp
  evaluators/PoroElasticityResidMass.hpp
  evaluators/PoroElasticityResidMass_Def.hpp
  evaluators/UnSatPoroElasticityResidMass.hpp
  evaluators/UnSatPoroElasticityResidMass_Def.hpp
  evaluators/TLPoroPlasticityResidMomentum.hpp
  evaluators/TLPoroPlasticityResidMomentum_Def.hpp
  evaluators/TLPoroPlasticityResidMass.hpp
  evaluators/TLPoroPlasticityResidMass_Def.hpp
  evaluators/ThermoPoroPlasticityResidMomentum.hpp
  evaluators/ThermoPoroPlasticityResidMomentum_Def.hpp
  evaluators/ThermoPoroPlasticityResidMass.hpp
  evaluators/ThermoPoroPlasticityResidMass_Def.hpp
  evaluators/ThermoPoroPlasticityResidEnergy.hpp
  evaluators/ThermoPoroPlasticityResidEnergy_Def.hpp
  evaluators/Porosity.hpp
  evaluators/Porosity_Def.hpp
  evaluators/KCPermeability.hpp
  evaluators/KCPermeability_Def.hpp
  evaluators/VanGenuchtenPermeability.hpp
  evaluators/VanGenuchtenPermeability_Def.hpp
  evaluators/VanGenuchtenSaturation.hpp
  evaluators/VanGenuchtenSaturation_Def.hpp
  evaluators/ThermoMechanicalEnergyResidual.hpp
  evaluators/ThermoMechanicalEnergyResidual_Def.hpp
  evaluators/ThermoMechanicalMomentumResidual.hpp
  evaluators/ThermoMechanicalMomentumResidual_Def.hpp
  evaluators/ThermoMechanicalStress.hpp
  evaluators/ThermoMechanicalStress_Def.hpp
  evaluators/Time.hpp
  evaluators/Time_Def.hpp
  evaluators/TimeDepBC.hpp
  evaluators/TimeDepBC_Def.hpp
  evaluators/TimeTracBC.hpp
  evaluators/TimeTracBC_Def.hpp
  evaluators/DiffusionCoefficient.hpp
  evaluators/DiffusionCoefficient_Def.hpp
  evaluators/EquilibriumConstant.hpp
  evaluators/EquilibriumConstant_Def.hpp
  evaluators/TrappedSolvent.hpp
  evaluators/TrappedSolvent_Def.hpp
  evaluators/TrappedConcentration.hpp
  evaluators/TrappedConcentration_Def.hpp
  evaluators/EffectiveDiffusivity.hpp
  evaluators/EffectiveDiffusivity_Def.hpp
  evaluators/TransportCoefficients.hpp
  evaluators/TransportCoefficients_Def.hpp
  evaluators/TotalConcentration.hpp
  evaluators/TotalConcentration_Def.hpp
  evaluators/HDiffusionDeformationMatterResidual.hpp
  evaluators/HDiffusionDeformationMatterResidual_Def.hpp
  evaluators/ScalarL2ProjectionResidual.hpp
  evaluators/ScalarL2ProjectionResidual_Def.hpp
  evaluators/L2ProjectionResidual.hpp
  evaluators/L2ProjectionResidual_Def.hpp
  evaluators/StrainRateFactor.hpp
  evaluators/StrainRateFactor_Def.hpp
  evaluators/TauContribution.hpp
  evaluators/TauContribution_Def.hpp
  evaluators/MixtureThermalExpansion.hpp
  evaluators/MixtureThermalExpansion_Def.hpp
  evaluators/MixtureSpecificHeat.hpp
  evaluators/MixtureSpecificHeat_Def.hpp
  evaluators/StabParameter.hpp
  evaluators/StabParameter_Def.hpp
  evaluators/GradientElementLength.hpp
  evaluators/GradientElementLength_Def.hpp
  evaluators/UnitGradient.hpp
  evaluators/UnitGradient_Def.hpp
  evaluators/SetField.hpp
  evaluators/SetField_Def.hpp
  evaluators/AssumedStrain.hpp
  evaluators/AssumedStrain_Def.hpp
  evaluators/MooneyRivlinDamage.hpp
  evaluators/MooneyRivlinDamage_Def.hpp
  evaluators/MooneyRivlin_Incompressible.hpp
  evaluators/MooneyRivlin_Incompressible_Def.hpp
  evaluators/MooneyRivlin_Incompressible_Damage.hpp
  evaluators/MooneyRivlin_Incompressible_Damage_Def.hpp
  evaluators/RecoveryModulus.hpp
  evaluators/RecoveryModulus_Def.hpp
  evaluators/FaceFractureCriteria.hpp
  evaluators/FaceFractureCriteria_Def.hpp
  evaluators/FaceAverage.hpp
  evaluators/FaceAverage_Def.hpp
  evaluators/AAA.hpp
  evaluators/AAA_Def.hpp
  evaluators/LatticeDefGrad.hpp
  evaluators/LatticeDefGrad_Def.hpp
  evaluators/TvergaardHutchinson.hpp
  evaluators/TvergaardHutchinson_Def.hpp
  evaluators/SurfaceCohesiveResidual.hpp
  evaluators/SurfaceCohesiveResidual_Def.hpp
  evaluators/ConstitutiveModelInterface.hpp
  evaluators/ConstitutiveModelInterface_Def.hpp
  evaluators/ConstitutiveModelParameters.hpp
  evaluators/ConstitutiveModelParameters_Def.hpp
  evaluators/Kinematics.hpp
  evaluators/Kinematics_Def.hpp
  evaluators/BifurcationCheck.hpp
  evaluators/BifurcationCheck_Def.hpp
  evaluators/TransportResidual.hpp
  evaluators/TransportResidual_Def.hpp
  evaluators/ThermoMechanicalCoefficients.hpp
  evaluators/ThermoMechanicalCoefficients_Def.hpp
  )

#LCM models
SET(SOURCES ${SOURCES}
  models/ConstitutiveModel.cpp
  models/NeohookeanModel.cpp
  models/J2Model.cpp
  models/AnisotropicHyperelasticDamageModel.cpp
  models/GursonModel.cpp
  models/GursonHMRModel.cpp
  models/MooneyRivlinModel.cpp
  models/RIHMRModel.cpp
  models/J2FiberModel.cpp
  models/AnisotropicDamageModel.cpp
  )

SET(HEADERS ${HEADERS}
  models/ConstitutiveModel.hpp
  models/ConstitutiveModel_Def.hpp
  models/NeohookeanModel.hpp
  models/NeohookeanModel_Def.hpp
  models/J2Model.hpp
  models/J2Model_Def.hpp
  models/AnisotropicHyperelasticDamageModel.hpp
  models/AnisotropicHyperelasticDamageModel_Def.hpp
  models/GursonModel.hpp
  models/GursonModel_Def.hpp
  models/GursonHMRModel.hpp
  models/GursonHMRModel_Def.hpp
  models/MooneyRivlinModel.hpp
  models/MooneyRivlinModel_Def.hpp
  models/RIHMRModel.hpp
  models/RIHMRModel_Def.hpp
  models/J2FiberModel.hpp
  models/J2FiberModel_Def.hpp
  models/AnisotropicDamageModel.hpp
  models/AnisotropicDamageModel_Def.hpp
  )

#LCM utils
SET(SOURCES ${SOURCES}
  utils/FloatingPoint.cc
  utils/Fracture.cc
  utils/Partition.cc
  utils/BarycentricSubdivision.cc
  utils/MinimumSurface.cc
  utils/Topology.cc
  utils/LocalNonlinearSolver.cc
  utils/Projection.cpp
  )

SET(HEADERS ${HEADERS}
  utils/FloatingPoint.h
  utils/Fracture.h
  utils/Partition.h
  utils/Topology.h
  utils/LocalNonlinearSolver.h
  utils/Projection.hpp
  )

IF(ALBANY_LAME OR ALBANY_LAMENT)
  SET(SOURCES ${SOURCES}
    utils/LameUtils.cpp
    problems/LameProblem.cpp
    )
  SET(HEADERS ${HEADERS}
    utils/LameUtils.hpp
    problems/LameProblem.hpp
    )
ENDIF()

IF(ALBANY_LAME)
  SET(SOURCES ${SOURCES}
    evaluators/LameStress.cpp
    )
  SET(HEADERS ${HEADERS}
    evaluators/LameStress.hpp
    evaluators/LameStress_Def.hpp
    )
ENDIF()

IF(ALBANY_LAMENT)
  SET(SOURCES ${SOURCES}
    evaluators/LamentStress.cpp
    )
  SET(HEADERS ${HEADERS}
    evaluators/LamentStress.hpp
    evaluators/LamentStress_Def.hpp
    )
ENDIF()

IF(ALBANY_PERIDIGM)
  SET(HEADERS ${HEADERS}
    utils/PeridigmManager.hpp
    )
  SET(SOURCES ${SOURCES}
    utils/PeridigmManager.cpp
    )
ENDIF()

include_directories (${Trilinos_INCLUDE_DIRS}
  ${Albany_SOURCE_DIR}/src
  ${Albany_SOURCE_DIR}/src/evaluators
  ${Albany_SOURCE_DIR}/src/problems 
  ${Albany_SOURCE_DIR}/src/responses
  ${Albany_SOURCE_DIR}/src/disc/stk
  ${Albany_SOURCE_DIR}/src/adapt
  ./evaluators
  ./utils
  ${ALBANY_FMDB_INCLUDE_DIRS} 
  ${Trilinos_TPL_INCLUDE_DIRS}
  )

link_directories (${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})

add_library(LCM ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})
set_target_properties(LCM PROPERTIES
  PUBLIC_HEADER "${HEADERS}")

IF ( INSTALL_ALBANY )
  install(TARGETS LCM EXPORT albany-export
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()

# Tests
#
# Many (all?) the tests have a dependency on stk. Disable them for now if
# using the RPI database
if(NOT ALBANY_SCOREC)
  add_executable(PartitionTest test/utils/PartitionTest.cc)
  add_executable(NodeUpdate test/utils/NodeUpdate.cc)
  add_executable(Subdivision test/utils/Subdivision.cc)
  add_executable(MinSurface test/utils/MinSurface.cc)
  add_executable(Test1_Subdivision test/utils/Test1_Subdivision.cc)
  add_executable(Test2_Subdivision test/utils/Test2_Subdivision.cc)
  add_executable(MeshComponents test/utils/MeshComponents.cc)
  add_executable(MaterialPointSimulator test/utils/MaterialPointSimulator.cc)
  add_executable(
    utLocalNonlinearSolver
    test/unit_tests/StandardUnitTestMain.cpp
    test/unit_tests/utLocalNonlinearSolver.cpp
    )
  add_executable(
    utSurfaceElement
    test/unit_tests/StandardUnitTestMain.cpp
    test/unit_tests/utSurfaceElement.cpp
    )
  
  IF (ALBANY_PERIDIGM)
    add_executable(
      utPeridigm
      utils/PeridigmManager.cpp
      test/unit_tests/StandardUnitTestMain.cpp
      test/unit_tests/utPeridigm.cpp
      )
    target_link_libraries(utPeridigm ${ALL_LIBRARIES})
  ENDIF()
  
  IF (ALBANY_LAME OR ALBANY_LAMENT)
    add_executable(utLameStress_elastic
      ${Albany_SOURCE_DIR}/src/PHAL_Dimension.cpp
      ${Albany_SOURCE_DIR}/src/PHAL_AlbanyTraits.cpp
      ${Albany_SOURCE_DIR}/src/Albany_Utils.cpp
      ${Albany_SOURCE_DIR}/src/Albany_StateManager.cpp
      ${Albany_SOURCE_DIR}/src/disc/stk/Albany_GenericSTKMeshStruct.cpp
      ${Albany_SOURCE_DIR}/src/disc/stk/Albany_GenericSTKFieldContainer.cpp
      ${Albany_SOURCE_DIR}/src/disc/stk/Albany_MultiSTKFieldContainer.cpp
      ${Albany_SOURCE_DIR}/src/disc/stk/Albany_OrdinarySTKFieldContainer.cpp
      ${Albany_SOURCE_DIR}/src/disc/stk/Albany_TmplSTKMeshStruct.cpp
      ${Albany_SOURCE_DIR}/src/disc/stk/Albany_STKDiscretization.cpp
      ${Albany_SOURCE_DIR}/src/problems/QCAD_MaterialDatabase.cpp
      utils/LameUtils.cpp
      evaluators/LameStress.cpp
      evaluators/SetField.cpp
      test/unit_tests/StandardUnitTestMain.cpp
      test/unit_tests/utLameStress_elastic.cpp
      )
    target_link_libraries(utLameStress_elastic ${LAME_LIB} ${LAMENT_LIB} ${ALB_TRILINOS_LIBS})
  ENDIF ()
  
  IF(BUILD_SHARED_LIBS)
    IF(ALBANY_SCOREC)
      SET(LCM_UT_LIBS albanyLib albanyAdapt fmdb albanySTK hydride LCM albanyLib)
    ELSE()
      SET(LCM_UT_LIBS albanyLib albanyAdapt albanySTK hydride LCM albanyLib)
    ENDIF()
  ELSE(BUILD_SHARED_LIBS)
    # We really don't link to anything in Hydride
    IF(ALBANY_SCOREC)
      SET(LCM_UT_LIBS LCM albanyAdapt albanyLib fmdb albanySTK albanyLib)
    ELSE()
      SET(LCM_UT_LIBS LCM albanyAdapt albanyLib albanySTK albanyLib )
    ENDIF()
  ENDIF(BUILD_SHARED_LIBS)
  
  
  #Linking tests
  target_link_libraries(PartitionTest ${LCM_UT_LIBS} ${ALL_LIBRARIES})
  target_link_libraries(NodeUpdate ${LCM_UT_LIBS} ${ALL_LIBRARIES})
  target_link_libraries(Subdivision ${LCM_UT_LIBS} ${ALL_LIBRARIES})
  target_link_libraries(MinSurface ${LCM_UT_LIBS} ${ALL_LIBRARIES})
  target_link_libraries(Test1_Subdivision ${LCM_UT_LIBS} ${ALL_LIBRARIES})
  target_link_libraries(Test2_Subdivision ${LCM_UT_LIBS} ${ALL_LIBRARIES})
  target_link_libraries(MeshComponents ${LCM_UT_LIBS} ${ALL_LIBRARIES})
  target_link_libraries(utLocalNonlinearSolver ${LCM_UT_LIBS} ${ALL_LIBRARIES})
  target_link_libraries(utSurfaceElement ${LCM_UT_LIBS} ${ALL_LIBRARIES})
  target_link_libraries(NodeUpdate ${LCM_UT_LIBS} ${ALL_LIBRARIES})
  target_link_libraries(MaterialPointSimulator ${LCM_UT_LIBS} ${ALL_LIBRARIES})
ENDIF(NOT ALBANY_SCOREC)
