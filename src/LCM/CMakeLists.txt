##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# LCM DIR
SET(LCM_DIR ${Albany_SOURCE_DIR}/src/LCM)

# LCM problems

# Glob source files for problems
file (GLOB problem-sources
  "${LCM_DIR}/problems/*.cpp"
)

# Glob header files for problems
file (GLOB problem-headers
  "${LCM_DIR}/problems/*.hpp"
)

SET(SOURCES )
SET(HEADERS )

# LCM evaluators
SET(SOURCES ${SOURCES}
  ${LCM_DIR}/evaluators/Stress.cpp
  ${LCM_DIR}/evaluators/CapExplicit.cpp
  ${LCM_DIR}/evaluators/CapImplicit.cpp
  ${LCM_DIR}/evaluators/Strain.cpp
  ${LCM_DIR}/evaluators/ElasticModulus.cpp
  ${LCM_DIR}/evaluators/PoissonsRatio.cpp
  ${LCM_DIR}/evaluators/DefGrad.cpp
  ${LCM_DIR}/evaluators/Neohookean.cpp
  ${LCM_DIR}/evaluators/KirchhoffStVenant.cpp
  ${LCM_DIR}/evaluators/J2Stress.cpp
  ${LCM_DIR}/evaluators/ElasticityResid.cpp
  ${LCM_DIR}/evaluators/TLElasResid.cpp
  ${LCM_DIR}/evaluators/MechanicsResidual.cpp
  ${LCM_DIR}/evaluators/HardeningModulus.cpp
  ${LCM_DIR}/evaluators/YieldStrength.cpp
  ${LCM_DIR}/evaluators/PisdWdF.cpp
  ${LCM_DIR}/evaluators/DamageResid.cpp
  ${LCM_DIR}/evaluators/J2Damage.cpp
  ${LCM_DIR}/evaluators/DamageLS.cpp
  ${LCM_DIR}/evaluators/SaturationModulus.cpp
  ${LCM_DIR}/evaluators/SaturationExponent.cpp
  ${LCM_DIR}/evaluators/CurrentCoords.cpp
  ${LCM_DIR}/evaluators/DamageSource.cpp
  ${LCM_DIR}/evaluators/BulkModulus.cpp
  ${LCM_DIR}/evaluators/ShearModulus.cpp
  ${LCM_DIR}/evaluators/DislocationDensity.cpp
  ${LCM_DIR}/evaluators/TotalStress.cpp
  ${LCM_DIR}/evaluators/BiotCoefficient.cpp
  ${LCM_DIR}/evaluators/BiotModulus.cpp
  ${LCM_DIR}/evaluators/PoroElasticityResidMomentum.cpp
  ${LCM_DIR}/evaluators/PoroElasticityResidMass.cpp
  ${LCM_DIR}/evaluators/UnSatPoroElasticityResidMass.cpp
  ${LCM_DIR}/evaluators/TLPoroPlasticityResidMomentum.cpp
  ${LCM_DIR}/evaluators/TLPoroPlasticityResidMass.cpp
  ${LCM_DIR}/evaluators/TLPoroStress.cpp
  ${LCM_DIR}/evaluators/ThermoPoroPlasticityResidMomentum.cpp
  ${LCM_DIR}/evaluators/ThermoPoroPlasticityResidMass.cpp
  ${LCM_DIR}/evaluators/ThermoPoroPlasticityResidEnergy.cpp
  ${LCM_DIR}/evaluators/Porosity.cpp
  ${LCM_DIR}/evaluators/KCPermeability.cpp
  ${LCM_DIR}/evaluators/VanGenuchtenPermeability.cpp
  ${LCM_DIR}/evaluators/VanGenuchtenSaturation.cpp
  ${LCM_DIR}/evaluators/ThermoMechanicalEnergyResidual.cpp
  ${LCM_DIR}/evaluators/ThermoMechanicalMomentumResidual.cpp
  ${LCM_DIR}/evaluators/ThermoMechanicalStress.cpp
  ${LCM_DIR}/evaluators/Time.cpp
  ${LCM_DIR}/evaluators/DiffusionCoefficient.cpp
  ${LCM_DIR}/evaluators/EffectiveDiffusivity.cpp
  ${LCM_DIR}/evaluators/EquilibriumConstant.cpp
  ${LCM_DIR}/evaluators/TrappedSolvent.cpp
  ${LCM_DIR}/evaluators/TrappedConcentration.cpp
  ${LCM_DIR}/evaluators/TransportCoefficients.cpp
  ${LCM_DIR}/evaluators/TotalConcentration.cpp
  ${LCM_DIR}/evaluators/HDiffusionDeformationMatterResidual.cpp
  ${LCM_DIR}/evaluators/ScalarL2ProjectionResidual.cpp
  ${LCM_DIR}/evaluators/L2ProjectionResidual.cpp
  ${LCM_DIR}/evaluators/TauContribution.cpp
  ${LCM_DIR}/evaluators/StrainRateFactor.cpp
  ${LCM_DIR}/evaluators/MixtureThermalExpansion.cpp
  ${LCM_DIR}/evaluators/MixtureSpecificHeat.cpp
  ${LCM_DIR}/evaluators/StabParameter.cpp
  ${LCM_DIR}/evaluators/GradientElementLength.cpp
  ${LCM_DIR}/evaluators/UnitGradient.cpp
  ${LCM_DIR}/evaluators/SetField.cpp
  ${LCM_DIR}/evaluators/AssumedStrain.cpp
  ${LCM_DIR}/evaluators/MooneyRivlinDamage.cpp
  ${LCM_DIR}/evaluators/MooneyRivlin_Incompressible.cpp
  ${LCM_DIR}/evaluators/MooneyRivlin_Incompressible_Damage.cpp
  ${LCM_DIR}/evaluators/RecoveryModulus.cpp
  ${LCM_DIR}/evaluators/FaceFractureCriteria.cpp
  ${LCM_DIR}/evaluators/FaceAverage.cpp
  ${LCM_DIR}/evaluators/AAA.cpp
  ${LCM_DIR}/evaluators/LatticeDefGrad.cpp
  ${LCM_DIR}/evaluators/TvergaardHutchinson.cpp
  ${LCM_DIR}/evaluators/Kinematics.cpp
  ${LCM_DIR}/evaluators/BifurcationCheck.cpp
  ${LCM_DIR}/evaluators/TransportResidual.cpp
  ${LCM_DIR}/evaluators/ThermoMechanicalCoefficients.cpp
  )

SET(HEADERS ${HEADERS}
  ${LCM_DIR}/evaluators/Stress.hpp
  ${LCM_DIR}/evaluators/Stress_Def.hpp
  ${LCM_DIR}/evaluators/CapExplicit.hpp
  ${LCM_DIR}/evaluators/CapExplicit_Def.hpp
  ${LCM_DIR}/evaluators/CapImplicit.hpp
  ${LCM_DIR}/evaluators/CapImplicit_Def.hpp
  ${LCM_DIR}/evaluators/Strain.hpp
  ${LCM_DIR}/evaluators/Strain_Def.hpp
  ${LCM_DIR}/evaluators/ElasticModulus.hpp
  ${LCM_DIR}/evaluators/ElasticModulus_Def.hpp
  ${LCM_DIR}/evaluators/PoissonsRatio.hpp
  ${LCM_DIR}/evaluators/PoissonsRatio_Def.hpp
  ${LCM_DIR}/evaluators/DefGrad.hpp
  ${LCM_DIR}/evaluators/DefGrad_Def.hpp
  ${LCM_DIR}/evaluators/Neohookean.hpp
  ${LCM_DIR}/evaluators/Neohookean_Def.hpp
  ${LCM_DIR}/evaluators/KirchhoffStVenant.hpp
  ${LCM_DIR}/evaluators/KirchhoffStVenant_Def.hpp
  ${LCM_DIR}/evaluators/J2Stress.hpp
  ${LCM_DIR}/evaluators/J2Stress_Def.hpp
  ${LCM_DIR}/evaluators/ElasticityResid.hpp
  ${LCM_DIR}/evaluators/ElasticityResid_Def.hpp
  ${LCM_DIR}/evaluators/TLElasResid.hpp
  ${LCM_DIR}/evaluators/TLElasResid_Def.hpp
  ${LCM_DIR}/evaluators/MechanicsResidual.hpp
  ${LCM_DIR}/evaluators/MechanicsResidual_Def.hpp
  ${LCM_DIR}/evaluators/HardeningModulus.hpp
  ${LCM_DIR}/evaluators/HardeningModulus_Def.hpp
  ${LCM_DIR}/evaluators/YieldStrength.hpp
  ${LCM_DIR}/evaluators/YieldStrength_Def.hpp
  ${LCM_DIR}/evaluators/PisdWdF.hpp
  ${LCM_DIR}/evaluators/PisdWdF_Def.hpp
  ${LCM_DIR}/evaluators/DamageResid.hpp
  ${LCM_DIR}/evaluators/DamageResid_Def.hpp
  ${LCM_DIR}/evaluators/J2Damage.hpp
  ${LCM_DIR}/evaluators/J2Damage_Def.hpp
  ${LCM_DIR}/evaluators/DamageLS.hpp
  ${LCM_DIR}/evaluators/DamageLS_Def.hpp
  ${LCM_DIR}/evaluators/SaturationModulus.hpp
  ${LCM_DIR}/evaluators/SaturationModulus_Def.hpp
  ${LCM_DIR}/evaluators/SaturationExponent.hpp
  ${LCM_DIR}/evaluators/SaturationExponent_Def.hpp
  ${LCM_DIR}/evaluators/CurrentCoords.hpp
  ${LCM_DIR}/evaluators/CurrentCoords_Def.hpp
  ${LCM_DIR}/evaluators/DamageSource.hpp
  ${LCM_DIR}/evaluators/DamageSource_Def.hpp
  ${LCM_DIR}/evaluators/BulkModulus.hpp
  ${LCM_DIR}/evaluators/BulkModulus_Def.hpp
  ${LCM_DIR}/evaluators/ShearModulus.hpp
  ${LCM_DIR}/evaluators/ShearModulus_Def.hpp
  ${LCM_DIR}/evaluators/DislocationDensity.hpp
  ${LCM_DIR}/evaluators/DislocationDensity_Def.hpp
  ${LCM_DIR}/evaluators/TotalStress.hpp
  ${LCM_DIR}/evaluators/TotalStress_Def.hpp
  ${LCM_DIR}/evaluators/TLPoroStress.hpp
  ${LCM_DIR}/evaluators/TLPoroStress_Def.hpp
  ${LCM_DIR}/evaluators/BiotCoefficient.hpp
  ${LCM_DIR}/evaluators/BiotCoefficient_Def.hpp
  ${LCM_DIR}/evaluators/BiotModulus.hpp
  ${LCM_DIR}/evaluators/BiotModulus_Def.hpp
  ${LCM_DIR}/evaluators/PoroElasticityResidMomentum.hpp
  ${LCM_DIR}/evaluators/PoroElasticityResidMomentum_Def.hpp
  ${LCM_DIR}/evaluators/PoroElasticityResidMass.hpp
  ${LCM_DIR}/evaluators/PoroElasticityResidMass_Def.hpp
  ${LCM_DIR}/evaluators/UnSatPoroElasticityResidMass.hpp
  ${LCM_DIR}/evaluators/UnSatPoroElasticityResidMass_Def.hpp
  ${LCM_DIR}/evaluators/TLPoroPlasticityResidMomentum.hpp
  ${LCM_DIR}/evaluators/TLPoroPlasticityResidMomentum_Def.hpp
  ${LCM_DIR}/evaluators/TLPoroPlasticityResidMass.hpp
  ${LCM_DIR}/evaluators/TLPoroPlasticityResidMass_Def.hpp
  ${LCM_DIR}/evaluators/ThermoPoroPlasticityResidMomentum.hpp
  ${LCM_DIR}/evaluators/ThermoPoroPlasticityResidMomentum_Def.hpp
  ${LCM_DIR}/evaluators/ThermoPoroPlasticityResidMass.hpp
  ${LCM_DIR}/evaluators/ThermoPoroPlasticityResidMass_Def.hpp
  ${LCM_DIR}/evaluators/ThermoPoroPlasticityResidEnergy.hpp
  ${LCM_DIR}/evaluators/ThermoPoroPlasticityResidEnergy_Def.hpp
  ${LCM_DIR}/evaluators/Porosity.hpp
  ${LCM_DIR}/evaluators/Porosity_Def.hpp
  ${LCM_DIR}/evaluators/KCPermeability.hpp
  ${LCM_DIR}/evaluators/KCPermeability_Def.hpp
  ${LCM_DIR}/evaluators/VanGenuchtenPermeability.hpp
  ${LCM_DIR}/evaluators/VanGenuchtenPermeability_Def.hpp
  ${LCM_DIR}/evaluators/VanGenuchtenSaturation.hpp
  ${LCM_DIR}/evaluators/VanGenuchtenSaturation_Def.hpp
  ${LCM_DIR}/evaluators/ThermoMechanicalEnergyResidual.hpp
  ${LCM_DIR}/evaluators/ThermoMechanicalEnergyResidual_Def.hpp
  ${LCM_DIR}/evaluators/ThermoMechanicalMomentumResidual.hpp
  ${LCM_DIR}/evaluators/ThermoMechanicalMomentumResidual_Def.hpp
  ${LCM_DIR}/evaluators/ThermoMechanicalStress.hpp
  ${LCM_DIR}/evaluators/ThermoMechanicalStress_Def.hpp
  ${LCM_DIR}/evaluators/Time.hpp
  ${LCM_DIR}/evaluators/Time_Def.hpp
  ${LCM_DIR}/evaluators/DiffusionCoefficient.hpp
  ${LCM_DIR}/evaluators/DiffusionCoefficient_Def.hpp
  ${LCM_DIR}/evaluators/EquilibriumConstant.hpp
  ${LCM_DIR}/evaluators/EquilibriumConstant_Def.hpp
  ${LCM_DIR}/evaluators/TrappedSolvent.hpp
  ${LCM_DIR}/evaluators/TrappedSolvent_Def.hpp
  ${LCM_DIR}/evaluators/TrappedConcentration.hpp
  ${LCM_DIR}/evaluators/TrappedConcentration_Def.hpp
  ${LCM_DIR}/evaluators/EffectiveDiffusivity.hpp
  ${LCM_DIR}/evaluators/EffectiveDiffusivity_Def.hpp
  ${LCM_DIR}/evaluators/TransportCoefficients.hpp
  ${LCM_DIR}/evaluators/TransportCoefficients_Def.hpp
  ${LCM_DIR}/evaluators/TotalConcentration.hpp
  ${LCM_DIR}/evaluators/TotalConcentration_Def.hpp
  ${LCM_DIR}/evaluators/HDiffusionDeformationMatterResidual.hpp
  ${LCM_DIR}/evaluators/HDiffusionDeformationMatterResidual_Def.hpp
  ${LCM_DIR}/evaluators/ScalarL2ProjectionResidual.hpp
  ${LCM_DIR}/evaluators/ScalarL2ProjectionResidual_Def.hpp
  ${LCM_DIR}/evaluators/L2ProjectionResidual.hpp
  ${LCM_DIR}/evaluators/L2ProjectionResidual_Def.hpp
  ${LCM_DIR}/evaluators/StrainRateFactor.hpp
  ${LCM_DIR}/evaluators/StrainRateFactor_Def.hpp
  ${LCM_DIR}/evaluators/TauContribution.hpp
  ${LCM_DIR}/evaluators/TauContribution_Def.hpp
  ${LCM_DIR}/evaluators/MixtureThermalExpansion.hpp
  ${LCM_DIR}/evaluators/MixtureThermalExpansion_Def.hpp
  ${LCM_DIR}/evaluators/MixtureSpecificHeat.hpp
  ${LCM_DIR}/evaluators/MixtureSpecificHeat_Def.hpp
  ${LCM_DIR}/evaluators/StabParameter.hpp
  ${LCM_DIR}/evaluators/StabParameter_Def.hpp
  ${LCM_DIR}/evaluators/GradientElementLength.hpp
  ${LCM_DIR}/evaluators/GradientElementLength_Def.hpp
  ${LCM_DIR}/evaluators/UnitGradient.hpp
  ${LCM_DIR}/evaluators/UnitGradient_Def.hpp
  ${LCM_DIR}/evaluators/SetField.hpp
  ${LCM_DIR}/evaluators/SetField_Def.hpp
  ${LCM_DIR}/evaluators/AssumedStrain.hpp
  ${LCM_DIR}/evaluators/AssumedStrain_Def.hpp
  ${LCM_DIR}/evaluators/MooneyRivlinDamage.hpp
  ${LCM_DIR}/evaluators/MooneyRivlinDamage_Def.hpp
  ${LCM_DIR}/evaluators/MooneyRivlin_Incompressible.hpp
  ${LCM_DIR}/evaluators/MooneyRivlin_Incompressible_Def.hpp
  ${LCM_DIR}/evaluators/MooneyRivlin_Incompressible_Damage.hpp
  ${LCM_DIR}/evaluators/MooneyRivlin_Incompressible_Damage_Def.hpp
  ${LCM_DIR}/evaluators/RecoveryModulus.hpp
  ${LCM_DIR}/evaluators/RecoveryModulus_Def.hpp
  ${LCM_DIR}/evaluators/FaceFractureCriteria.hpp
  ${LCM_DIR}/evaluators/FaceFractureCriteria_Def.hpp
  ${LCM_DIR}/evaluators/FaceAverage.hpp
  ${LCM_DIR}/evaluators/FaceAverage_Def.hpp
  ${LCM_DIR}/evaluators/AAA.hpp
  ${LCM_DIR}/evaluators/AAA_Def.hpp
  ${LCM_DIR}/evaluators/LatticeDefGrad.hpp
  ${LCM_DIR}/evaluators/LatticeDefGrad_Def.hpp
  ${LCM_DIR}/evaluators/TvergaardHutchinson.hpp
  ${LCM_DIR}/evaluators/TvergaardHutchinson_Def.hpp
  ${LCM_DIR}/evaluators/Kinematics.hpp
  ${LCM_DIR}/evaluators/Kinematics_Def.hpp
  ${LCM_DIR}/evaluators/BifurcationCheck.hpp
  ${LCM_DIR}/evaluators/BifurcationCheck_Def.hpp
  ${LCM_DIR}/evaluators/TransportResidual.hpp
  ${LCM_DIR}/evaluators/TransportResidual_Def.hpp
  ${LCM_DIR}/evaluators/ThermoMechanicalCoefficients.hpp
  ${LCM_DIR}/evaluators/ThermoMechanicalCoefficients_Def.hpp
  )

# LCM surface element evaluators
SET(SOURCES ${SOURCES}
  ${LCM_DIR}/evaluators/surface-element/SurfaceBasis.cpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceScalarJump.cpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceScalarGradient.cpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceScalarGradientOperator.cpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceDiffusionResidual.cpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceTLPoroMassResidual.cpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceHDiffusionDefResidual.cpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceL2ProjectionResidual.cpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceVectorJump.cpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceVectorGradient.cpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceVectorResidual.cpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceCohesiveResidual.cpp
)
SET(HEADERS ${HEADERS}
  ${LCM_DIR}/evaluators/surface-element/SurfaceBasis.hpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceBasis_Def.hpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceScalarJump.hpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceScalarJump_Def.hpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceVectorJump.hpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceVectorJump_Def.hpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceVectorGradient.hpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceVectorGradient_Def.hpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceScalarGradient.hpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceScalarGradient_Def.hpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceScalarGradientOperator.hpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceScalarGradientOperator_Def.hpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceDiffusionResidual.hpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceDiffusionResidual_Def.hpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceTLPoroMassResidual.hpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceTLPoroMassResidual_Def.hpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceHDiffusionDefResidual.hpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceHDiffusionDefResidual_Def.hpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceL2ProjectionResidual.hpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceL2ProjectionResidual_Def.hpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceVectorResidual.hpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceVectorResidual_Def.hpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceCohesiveResidual.hpp
  ${LCM_DIR}/evaluators/surface-element/SurfaceCohesiveResidual_Def.hpp
)

# LCM special boundary conditions
SET(SOURCES ${SOURCES}
  ${LCM_DIR}/evaluators/bc/KfieldBC.cpp
  ${LCM_DIR}/evaluators/bc/TorsionBC.cpp
  ${LCM_DIR}/evaluators/bc/TimeDepBC.cpp
  ${LCM_DIR}/evaluators/bc/TimeTracBC.cpp
)
SET(HEADERS ${HEADERS}
  ${LCM_DIR}/evaluators/bc/KfieldBC.hpp
  ${LCM_DIR}/evaluators/bc/KfieldBC_Def.hpp
  ${LCM_DIR}/evaluators/bc/TorsionBC.hpp
  ${LCM_DIR}/evaluators/bc/TorsionBC_Def.hpp
  ${LCM_DIR}/evaluators/bc/TimeDepBC.hpp
  ${LCM_DIR}/evaluators/bc/TimeDepBC_Def.hpp
  ${LCM_DIR}/evaluators/bc/TimeTracBC.hpp
  ${LCM_DIR}/evaluators/bc/TimeTracBC_Def.hpp
)

#LCM models
SET(SOURCES ${SOURCES}
  ${LCM_DIR}/models/ConstitutiveModelInterface.cpp
  ${LCM_DIR}/models/ConstitutiveModelParameters.cpp
  ${LCM_DIR}/models/ConstitutiveModel.cpp
  ${LCM_DIR}/models/NeohookeanModel.cpp
  ${LCM_DIR}/models/J2Model.cpp
  ${LCM_DIR}/models/AnisotropicHyperelasticDamageModel.cpp
  ${LCM_DIR}/models/GursonModel.cpp
  ${LCM_DIR}/models/GursonHMRModel.cpp
  ${LCM_DIR}/models/MooneyRivlinModel.cpp
  ${LCM_DIR}/models/RIHMRModel.cpp
  ${LCM_DIR}/models/J2FiberModel.cpp
  ${LCM_DIR}/models/AnisotropicDamageModel.cpp
  ${LCM_DIR}/models/ElasticDamageModel.cpp
  )

SET(HEADERS ${HEADERS}
  ${LCM_DIR}/models/ConstitutiveModelInterface.hpp
  ${LCM_DIR}/models/ConstitutiveModelInterface_Def.hpp
  ${LCM_DIR}/models/ConstitutiveModelParameters.hpp
  ${LCM_DIR}/models/ConstitutiveModelParameters_Def.hpp
  ${LCM_DIR}/models/ConstitutiveModel.hpp
  ${LCM_DIR}/models/ConstitutiveModel_Def.hpp
  ${LCM_DIR}/models/NeohookeanModel.hpp
  ${LCM_DIR}/models/NeohookeanModel_Def.hpp
  ${LCM_DIR}/models/J2Model.hpp
  ${LCM_DIR}/models/J2Model_Def.hpp
  ${LCM_DIR}/models/AnisotropicHyperelasticDamageModel.hpp
  ${LCM_DIR}/models/AnisotropicHyperelasticDamageModel_Def.hpp
  ${LCM_DIR}/models/GursonModel.hpp
  ${LCM_DIR}/models/GursonModel_Def.hpp
  ${LCM_DIR}/models/GursonHMRModel.hpp
  ${LCM_DIR}/models/GursonHMRModel_Def.hpp
  ${LCM_DIR}/models/MooneyRivlinModel.hpp
  ${LCM_DIR}/models/MooneyRivlinModel_Def.hpp
  ${LCM_DIR}/models/RIHMRModel.hpp
  ${LCM_DIR}/models/RIHMRModel_Def.hpp
  ${LCM_DIR}/models/J2FiberModel.hpp
  ${LCM_DIR}/models/J2FiberModel_Def.hpp
  ${LCM_DIR}/models/AnisotropicDamageModel.hpp
  ${LCM_DIR}/models/AnisotropicDamageModel_Def.hpp
  ${LCM_DIR}/models/ElasticDamageModel.hpp
  ${LCM_DIR}/models/ElasticDamageModel_Def.hpp
  )

#LCM utils
SET(SOURCES ${SOURCES}
  ${LCM_DIR}/utils/FloatingPoint.cc
  ${LCM_DIR}/utils/Partition.cc
  ${LCM_DIR}/utils/LocalNonlinearSolver.cc
  ${LCM_DIR}/utils/Projection.cpp
  )

SET(HEADERS ${HEADERS}
  ${LCM_DIR}/utils/FloatingPoint.h
  ${LCM_DIR}/utils/Partition.h
  ${LCM_DIR}/utils/LocalNonlinearSolver.h
  ${LCM_DIR}/utils/Projection.hpp
  )

#LCM utils/topology
SET(SOURCES ${SOURCES}
  ${LCM_DIR}/utils/topology/BarycentricSubdivision.cc
  ${LCM_DIR}/utils/topology/MinimumSurface.cc
  ${LCM_DIR}/utils/topology/Topology.cc
  ${LCM_DIR}/utils/topology/Topology_FractureCriterion.cc
  )

SET(HEADERS ${HEADERS}
  ${LCM_DIR}/utils/topology/Topology.h
  ${LCM_DIR}/utils/topology/Topology_FractureCriterion.h
  ${LCM_DIR}/utils/topology/Topology_Types.h
  )

IF(ALBANY_LAME OR ALBANY_LAMENT)
  SET(SOURCES ${SOURCES}
    ${LCM_DIR}/utils/LameUtils.cpp
    ${LCM_DIR}/problems/lame/LameProblem.cpp
    )
  SET(HEADERS ${HEADERS}
    ${LCM_DIR}/utils/LameUtils.hpp
    ${LCM_DIR}/problems/lame/LameProblem.hpp
    )
ENDIF()

IF(ALBANY_LAME)
  SET(SOURCES ${SOURCES}
    ${LCM_DIR}/evaluators/lame/LameStress.cpp
    )
  SET(HEADERS ${HEADERS}
    ${LCM_DIR}/evaluators/lame/LameStress.hpp
    ${LCM_DIR}/evaluators/lame/LameStress_Def.hpp
    )
ENDIF()

IF(ALBANY_LAMENT)
  SET(SOURCES ${SOURCES}
    ${LCM_DIR}/evaluators/lame/LamentStress.cpp
    )
  SET(HEADERS ${HEADERS}
    ${LCM_DIR}/evaluators/lame/LamentStress.hpp
    ${LCM_DIR}/evaluators/lame/LamentStress_Def.hpp
    )
ENDIF()

IF(ALBANY_PERIDIGM)
  SET(HEADERS ${HEADERS}
    ${LCM_DIR}/utils/PeridigmManager.hpp
    )
  SET(SOURCES ${SOURCES}
    ${LCM_DIR}/utils/PeridigmManager.cpp
    )
ENDIF()

set(SOURCES ${SOURCES}
  ${problem-sources}
)

set(HEADERS ${HEADERS}
  ${problem-headers}
)

set(INCLUDE_DIRS
  ${Trilinos_INCLUDE_DIRS}
  ${Albany_SOURCE_DIR}/src
  ${Albany_SOURCE_DIR}/src/evaluators
  ${Albany_SOURCE_DIR}/src/problems 
  ${Albany_SOURCE_DIR}/src/responses
  ${Albany_SOURCE_DIR}/src/disc/stk
  ${Albany_SOURCE_DIR}/src/adapt
  ${LCM_DIR}/problems
  ${LCM_DIR}/evaluators
  ${LCM_DIR}/evaluators/bc
  ${LCM_DIR}/evaluators/surface-element
  ${LCM_DIR}/models
  ${LCM_DIR}/utils
  ${LCM_DIR}/utils/topology
  ${ALBANY_FMDB_INCLUDE_DIRS} 
  ${Trilinos_TPL_INCLUDE_DIRS}
)

#message("include dirs - ${INCLUDE_DIRS}")

include_directories(${INCLUDE_DIRS})

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
