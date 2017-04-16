##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//
#
# LCM DIR
SET(LCM_DIR ${Albany_SOURCE_DIR}/src/LCM)

# LCM problems

set(problem-sources
  "${LCM_DIR}/problems/ConstitutiveDriverProblem.cpp"
  "${LCM_DIR}/problems/ElasticityProblem.cpp"
  "${LCM_DIR}/problems/ElectroMechanicsProblem.cpp"
  "${LCM_DIR}/problems/FieldNameMap.cpp"
  "${LCM_DIR}/problems/HMCProblem.cpp"
  "${LCM_DIR}/problems/MechanicsProblem.cpp"
  "${LCM_DIR}/problems/ThermoElasticityProblem.cpp"
)
set(problem-headers
  "${LCM_DIR}/problems/ConstitutiveDriverProblem.hpp"
  "${LCM_DIR}/problems/ElasticityProblem.hpp"
  "${LCM_DIR}/problems/ElectroMechanicsProblem.hpp"
  "${LCM_DIR}/problems/FieldNameMap.hpp"
  "${LCM_DIR}/problems/HMCProblem.hpp"
  "${LCM_DIR}/problems/MechanicsProblem.hpp"
  "${LCM_DIR}/problems/ThermoElasticityProblem.hpp"
)
IF (ALBANY_PERIDIGM AND ALBANY_EPETRA)
  set(problem-sources ${problem-sources}
    "${LCM_DIR}/problems/PeridigmProblem.cpp"
  )
  set(problem-headers ${problem-headers}
    "${LCM_DIR}/problems/PeridigmProblem.hpp"
  )
ENDIF()

SET(SOURCES )
SET(HEADERS )

# LCM evaluators
set(evaluator-sources
  "${LCM_DIR}/evaluators/BifurcationCheck.cpp"
  "${LCM_DIR}/evaluators/Density.cpp"
  "${LCM_DIR}/evaluators/NodePointVecInterpolation.cpp"
  "${LCM_DIR}/evaluators/SetField.cpp"
  "${LCM_DIR}/evaluators/Time.cpp"
  "${LCM_DIR}/parallel_evaluators/ParallelSetField.cpp"
)
set(evaluator-headers
  "${LCM_DIR}/evaluators/BifurcationCheck_Def.hpp"
  "${LCM_DIR}/evaluators/BifurcationCheck.hpp"
  "${LCM_DIR}/evaluators/Density_Def.hpp"
  "${LCM_DIR}/evaluators/Density.hpp"
  "${LCM_DIR}/evaluators/NodePointVecInterpolation_Def.hpp"
  "${LCM_DIR}/evaluators/NodePointVecInterpolation.hpp"
  "${LCM_DIR}/evaluators/SetField_Def.hpp"
  "${LCM_DIR}/evaluators/SetField.hpp"
  "${LCM_DIR}/evaluators/Time_Def.hpp"
  "${LCM_DIR}/evaluators/Time.hpp"
  "${LCM_DIR}/parallel_evaluators/ParallelSetField_Def.hpp"
  "${LCM_DIR}/parallel_evaluators/ParallelSetField.hpp"
)

#LCM model evaluators 
set(model-eval-sources
  "${LCM_DIR}/Schwarz_Alternating.cpp"
  "${LCM_DIR}/Schwarz_ObserverImpl.cpp"
  "${LCM_DIR}/Schwarz_PiroObserver.cpp"
  "${LCM_DIR}/Schwarz_StatelessObserverImpl.cpp"
)
set(model-eval-headers
  "${LCM_DIR}/Schwarz_Alternating.hpp"
  "${LCM_DIR}/Schwarz_ObserverImpl.hpp"
  "${LCM_DIR}/Schwarz_PiroObserver.hpp"
  "${LCM_DIR}/Schwarz_StatelessObserverImpl.hpp"
)
IF (ALBANY_HAVE_STK)
  set(model-eval-sources ${model-eval-sources}
    "${LCM_DIR}/Schwarz_BoundaryJacobian.cpp"
    "${LCM_DIR}/Schwarz_CoupledJacobian.cpp"
    "${LCM_DIR}/Schwarz_Coupled.cpp"
  )
  set(model-eval-headers ${model-eval-headers}
    "${LCM_DIR}/Schwarz_BoundaryJacobian.hpp"
    "${LCM_DIR}/Schwarz_CoupledJacobian.hpp"
    "${LCM_DIR}/Schwarz_Coupled.hpp"
  )
ENDIF(ALBANY_HAVE_STK)

# LCM special boundary conditions
set(bc-sources
  "${LCM_DIR}/evaluators/bc/EquilibriumConcentrationBC.cpp"
  "${LCM_DIR}/evaluators/bc/KfieldBC.cpp"
  "${LCM_DIR}/evaluators/bc/RigidContactBC.cpp"
  "${LCM_DIR}/evaluators/bc/TimeDepBC.cpp"
  "${LCM_DIR}/evaluators/bc/TimeTracBC.cpp"
  "${LCM_DIR}/evaluators/bc/TorsionBC.cpp"
)
set(bc-headers
  "${LCM_DIR}/evaluators/bc/EquilibriumConcentrationBC_Def.hpp"
  "${LCM_DIR}/evaluators/bc/EquilibriumConcentrationBC.hpp"
  "${LCM_DIR}/evaluators/bc/KfieldBC_Def.hpp"
  "${LCM_DIR}/evaluators/bc/KfieldBC.hpp"
  "${LCM_DIR}/evaluators/bc/RigidContactBC_Def.hpp"
  "${LCM_DIR}/evaluators/bc/RigidContactBC.hpp"
  "${LCM_DIR}/evaluators/bc/TimeDepBC_Def.hpp"
  "${LCM_DIR}/evaluators/bc/TimeDepBC.hpp"
  "${LCM_DIR}/evaluators/bc/TimeTracBC_Def.hpp"
  "${LCM_DIR}/evaluators/bc/TimeTracBC.hpp"
  "${LCM_DIR}/evaluators/bc/TorsionBC_Def.hpp"
  "${LCM_DIR}/evaluators/bc/TorsionBC.hpp"
)
IF (ALBANY_HAVE_STK)
  set(bc-sources ${bc-sources}
    "${LCM_DIR}/evaluators/bc/SchwarzBC.cpp"
    "${LCM_DIR}/evaluators/bc/PDNeighborFitBC.cpp"
  )
  set(bc-headers ${bc-headers}
    "${LCM_DIR}/evaluators/bc/SchwarzBC_Def.hpp"
    "${LCM_DIR}/evaluators/bc/SchwarzBC.hpp"
    "${LCM_DIR}/evaluators/bc/PDNeighborFitBC_Def.hpp"
    "${LCM_DIR}/evaluators/bc/PDNeighborFitBC.hpp"
  )
ENDIF (ALBANY_HAVE_STK)

# LCM contact
IF (ALBANY_CONTACT)
  set(contact-sources
    "${LCM_DIR}/evaluators/contact/MortarContactConstraints.cpp"
    "${LCM_DIR}/evaluators/contact/MortarContactResidual.cpp"
  )
  set(contact-headers
    "${LCM_DIR}/evaluators/contact/MortarContactConstraints_Def.hpp"
    "${LCM_DIR}/evaluators/contact/MortarContactConstraints.hpp"
    "${LCM_DIR}/evaluators/contact/MortarContactResidual_Def.hpp"
    "${LCM_DIR}/evaluators/contact/MortarContactResidual.hpp"
  )
ENDIF (ALBANY_CONTACT)

# LCM HMC evaluators
set(hmc-sources
  "${LCM_DIR}/evaluators/HMC/HMC_MicroResidual.cpp"
  "${LCM_DIR}/evaluators/HMC/HMC_StrainDifference.cpp"
  "${LCM_DIR}/evaluators/HMC/HMC_Stresses.cpp"
  "${LCM_DIR}/evaluators/HMC/HMC_TotalStress.cpp"
  "${LCM_DIR}/evaluators/HMC/UpdateField.cpp"
)
set(hmc-headers
  "${LCM_DIR}/evaluators/HMC/HMC_MicroResidual_Def.hpp"
  "${LCM_DIR}/evaluators/HMC/HMC_MicroResidual.hpp"
  "${LCM_DIR}/evaluators/HMC/HMC_StrainDifference_Def.hpp"
  "${LCM_DIR}/evaluators/HMC/HMC_StrainDifference.hpp"
  "${LCM_DIR}/evaluators/HMC/HMC_Stresses_Def.hpp"
  "${LCM_DIR}/evaluators/HMC/HMC_Stresses.hpp"
  "${LCM_DIR}/evaluators/HMC/HMC_TotalStress_Def.hpp"
  "${LCM_DIR}/evaluators/HMC/HMC_TotalStress.hpp"
  "${LCM_DIR}/evaluators/HMC/UpdateField_Def.hpp"
  "${LCM_DIR}/evaluators/HMC/UpdateField.hpp"
)

# LCM kinematics evaluators
set(kin-sources
  "${LCM_DIR}/evaluators/kinematics/CurrentCoords.cpp"
  "${LCM_DIR}/evaluators/kinematics/DefGrad.cpp"
  "${LCM_DIR}/evaluators/kinematics/FirstPK.cpp"
  "${LCM_DIR}/evaluators/kinematics/Kinematics.cpp"
  "${LCM_DIR}/evaluators/kinematics/LatticeDefGrad.cpp"
)
set(kin-headers
  "${LCM_DIR}/evaluators/kinematics/CurrentCoords_Def.hpp"
  "${LCM_DIR}/evaluators/kinematics/CurrentCoords.hpp"
  "${LCM_DIR}/evaluators/kinematics/DefGrad_Def.hpp"
  "${LCM_DIR}/evaluators/kinematics/DefGrad.hpp"
  "${LCM_DIR}/evaluators/kinematics/FirstPK_Def.hpp"
  "${LCM_DIR}/evaluators/kinematics/FirstPK.hpp"
  "${LCM_DIR}/evaluators/kinematics/Kinematics_Def.hpp"
  "${LCM_DIR}/evaluators/kinematics/Kinematics.hpp"
  "${LCM_DIR}/evaluators/kinematics/LatticeDefGrad_Def.hpp"
  "${LCM_DIR}/evaluators/kinematics/LatticeDefGrad.hpp"
)

# LCM linear elasticity evaluators
set(lin-sources
  "${LCM_DIR}/evaluators/linear-elasticity/ElasticModulus.cpp"
  "${LCM_DIR}/evaluators/linear-elasticity/PoissonsRatio.cpp"
  "${LCM_DIR}/evaluators/linear-elasticity/Strain.cpp"
  "${LCM_DIR}/evaluators/linear-elasticity/Stress.cpp"
)
set(lin-headers
  "${LCM_DIR}/evaluators/linear-elasticity/ElasticModulus_Def.hpp"
  "${LCM_DIR}/evaluators/linear-elasticity/ElasticModulus.hpp"
  "${LCM_DIR}/evaluators/linear-elasticity/PoissonsRatio_Def.hpp"
  "${LCM_DIR}/evaluators/linear-elasticity/PoissonsRatio.hpp"
  "${LCM_DIR}/evaluators/linear-elasticity/Strain_Def.hpp"
  "${LCM_DIR}/evaluators/linear-elasticity/Strain.hpp"
  "${LCM_DIR}/evaluators/linear-elasticity/Stress_Def.hpp"
  "${LCM_DIR}/evaluators/linear-elasticity/Stress.hpp"
)

# LCM poro evaluators
set(poro-sources
  "${LCM_DIR}/evaluators/poro/BiotCoefficient.cpp"
  "${LCM_DIR}/evaluators/poro/BiotModulus.cpp"
  "${LCM_DIR}/evaluators/poro/KCPermeability.cpp"
  "${LCM_DIR}/evaluators/poro/MixtureSpecificHeat.cpp"
  "${LCM_DIR}/evaluators/poro/MixtureThermalExpansion.cpp"
  "${LCM_DIR}/evaluators/poro/Porosity.cpp"
  "${LCM_DIR}/evaluators/poro/ThermoPoroPlasticityResidEnergy.cpp"
  "${LCM_DIR}/evaluators/poro/ThermoPoroPlasticityResidMass.cpp"
  "${LCM_DIR}/evaluators/poro/TLPoroPlasticityResidMass.cpp"
  "${LCM_DIR}/evaluators/poro/TLPoroPlasticityResidMomentum.cpp"
  "${LCM_DIR}/evaluators/poro/TLPoroStress.cpp"
)
set(poro-headers
  "${LCM_DIR}/evaluators/poro/BiotCoefficient_Def.hpp"
  "${LCM_DIR}/evaluators/poro/BiotCoefficient.hpp"
  "${LCM_DIR}/evaluators/poro/BiotModulus_Def.hpp"
  "${LCM_DIR}/evaluators/poro/BiotModulus.hpp"
  "${LCM_DIR}/evaluators/poro/KCPermeability_Def.hpp"
  "${LCM_DIR}/evaluators/poro/KCPermeability.hpp"
  "${LCM_DIR}/evaluators/poro/MixtureSpecificHeat_Def.hpp"
  "${LCM_DIR}/evaluators/poro/MixtureSpecificHeat.hpp"
  "${LCM_DIR}/evaluators/poro/MixtureThermalExpansion_Def.hpp"
  "${LCM_DIR}/evaluators/poro/MixtureThermalExpansion.hpp"
  "${LCM_DIR}/evaluators/poro/Porosity_Def.hpp"
  "${LCM_DIR}/evaluators/poro/Porosity.hpp"
  "${LCM_DIR}/evaluators/poro/ThermoPoroPlasticityResidEnergy_Def.hpp"
  "${LCM_DIR}/evaluators/poro/ThermoPoroPlasticityResidEnergy.hpp"
  "${LCM_DIR}/evaluators/poro/ThermoPoroPlasticityResidMass_Def.hpp"
  "${LCM_DIR}/evaluators/poro/ThermoPoroPlasticityResidMass.hpp"
  "${LCM_DIR}/evaluators/poro/TLPoroPlasticityResidMass_Def.hpp"
  "${LCM_DIR}/evaluators/poro/TLPoroPlasticityResidMass.hpp"
  "${LCM_DIR}/evaluators/poro/TLPoroPlasticityResidMomentum_Def.hpp"
  "${LCM_DIR}/evaluators/poro/TLPoroPlasticityResidMomentum.hpp"
  "${LCM_DIR}/evaluators/poro/TLPoroStress_Def.hpp"
  "${LCM_DIR}/evaluators/poro/TLPoroStress.hpp"
)

# LCM peridigm evaluators
IF (ALBANY_PERIDIGM AND ALBANY_EPETRA)
  set(pd-sources
    "${LCM_DIR}/evaluators/peridigm/AlbanyPeridigmOBCFunctional.cpp"
    "${LCM_DIR}/evaluators/peridigm/GatherSphereVolume.cpp"
    "${LCM_DIR}/evaluators/peridigm/PeridigmForce.cpp"
    "${LCM_DIR}/evaluators/peridigm/PeridigmPartialStress.cpp"
  )
  set(pd-headers
    "${LCM_DIR}/evaluators/peridigm/AlbanyPeridigmOBCFunctional.hpp"
    "${LCM_DIR}/evaluators/peridigm/GatherSphereVolume_Def.hpp"
    "${LCM_DIR}/evaluators/peridigm/GatherSphereVolume.hpp"
    "${LCM_DIR}/evaluators/peridigm/PeridigmForce_Def.hpp"
    "${LCM_DIR}/evaluators/peridigm/PeridigmForce.hpp"
    "${LCM_DIR}/evaluators/peridigm/PeridigmPartialStress_Def.hpp"
    "${LCM_DIR}/evaluators/peridigm/PeridigmPartialStress.hpp"
  )
ENDIF (ALBANY_PERIDIGM AND ALBANY_EPETRA)

# LCM projection evaluators
set(proj-sources
  "${LCM_DIR}/evaluators/projection/ScalarL2ProjectionResidual.cpp"
)
set(proj-headers
  "${LCM_DIR}/evaluators/projection/ScalarL2ProjectionResidual_Def.hpp"
  "${LCM_DIR}/evaluators/projection/ScalarL2ProjectionResidual.hpp"
)

# LCM residual evaluators
set(residual-sources
  "${LCM_DIR}/evaluators/residuals/ElasticityResid.cpp"
  "${LCM_DIR}/evaluators/residuals/ElectrostaticResidual.cpp"
  "${LCM_DIR}/evaluators/residuals/HDiffusionDeformationMatterResidual.cpp"
  "${LCM_DIR}/evaluators/residuals/MechanicsResidual.cpp"
  "${LCM_DIR}/evaluators/residuals/StabilizedPressureResidual.cpp"
  "${LCM_DIR}/evaluators/residuals/TLElasResid.cpp"
)
set(residual-headers
  "${LCM_DIR}/evaluators/residuals/ElasticityResid_Def.hpp"
  "${LCM_DIR}/evaluators/residuals/ElasticityResid.hpp"
  "${LCM_DIR}/evaluators/residuals/ElectrostaticResidual_Def.hpp"
  "${LCM_DIR}/evaluators/residuals/ElectrostaticResidual.hpp"
  "${LCM_DIR}/evaluators/residuals/HDiffusionDeformationMatterResidual_Def.hpp"
  "${LCM_DIR}/evaluators/residuals/HDiffusionDeformationMatterResidual.hpp"
  "${LCM_DIR}/evaluators/residuals/MechanicsResidual_Def.hpp"
  "${LCM_DIR}/evaluators/residuals/MechanicsResidual.hpp"
  "${LCM_DIR}/evaluators/residuals/StabilizedPressureResidual_Def.hpp"
  "${LCM_DIR}/evaluators/residuals/StabilizedPressureResidual.hpp"
  "${LCM_DIR}/evaluators/residuals/TLElasResid_Def.hpp"
  "${LCM_DIR}/evaluators/residuals/TLElasResid.hpp"
)

# LCM response evaluators
set(response-sources
  "${LCM_DIR}/evaluators/responses/IPtoNodalField.cpp"
  "${LCM_DIR}/evaluators/responses/ProjectIPtoNodalField.cpp"
)
set(response-headers
  "${LCM_DIR}/evaluators/responses/IPtoNodalField_Def.hpp"
  "${LCM_DIR}/evaluators/responses/IPtoNodalField.hpp"
  "${LCM_DIR}/evaluators/responses/ProjectIPtoNodalField_Def.hpp"
  "${LCM_DIR}/evaluators/responses/ProjectIPtoNodalField.hpp"
)

# LCM mesh size field evaluators
set(sf-sources
  "${LCM_DIR}/evaluators/size-field/MeshSizeField.cpp"
)
set(sf-headers
  "${LCM_DIR}/evaluators/size-field/MeshSizeField_Def.hpp"
  "${LCM_DIR}/evaluators/size-field/MeshSizeField.hpp"
)

# LCM surface element evaluators
set(surface-element-sources
  "${LCM_DIR}/evaluators/surface-element/SurfaceBasis.cpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceCohesiveResidual.cpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceDiffusionResidual.cpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceHDiffusionDefResidual.cpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceL2ProjectionResidual.cpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceScalarGradient.cpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceScalarGradientOperator.cpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceScalarJump.cpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceTLPoroMassResidual.cpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceVectorGradient.cpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceVectorJump.cpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceVectorResidual.cpp"
)
set(surface-element-headers
  "${LCM_DIR}/evaluators/surface-element/SurfaceBasis_Def.hpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceBasis.hpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceCohesiveResidual_Def.hpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceCohesiveResidual.hpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceDiffusionResidual_Def.hpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceDiffusionResidual.hpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceHDiffusionDefResidual_Def.hpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceHDiffusionDefResidual.hpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceL2ProjectionResidual_Def.hpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceL2ProjectionResidual.hpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceScalarGradient_Def.hpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceScalarGradient.hpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceScalarGradientOperator_Def.hpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceScalarGradientOperator.hpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceScalarJump_Def.hpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceScalarJump.hpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceTLPoroMassResidual_Def.hpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceTLPoroMassResidual.hpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceVectorGradient_Def.hpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceVectorGradient.hpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceVectorJump_Def.hpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceVectorJump.hpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceVectorResidual_Def.hpp"
  "${LCM_DIR}/evaluators/surface-element/SurfaceVectorResidual.hpp"
)

# LCM transport evaluators
set(transport-sources
  "${LCM_DIR}/evaluators/transport/DamageCoefficients.cpp"
  "${LCM_DIR}/evaluators/transport/GradientElementLength.cpp"
  "${LCM_DIR}/evaluators/transport/HeliumODEs.cpp"
  "${LCM_DIR}/evaluators/transport/ThermoMechanicalCoefficients.cpp"
  "${LCM_DIR}/evaluators/transport/TransportCoefficients.cpp"
  "${LCM_DIR}/evaluators/transport/TransportResidual.cpp"
)
set(transport-headers
  "${LCM_DIR}/evaluators/transport/DamageCoefficients_Def.hpp"
  "${LCM_DIR}/evaluators/transport/DamageCoefficients.hpp"
  "${LCM_DIR}/evaluators/transport/GradientElementLength_Def.hpp"
  "${LCM_DIR}/evaluators/transport/GradientElementLength.hpp"
  "${LCM_DIR}/evaluators/transport/HeliumODEs_Def.hpp"
  "${LCM_DIR}/evaluators/transport/HeliumODEs.hpp"
  "${LCM_DIR}/evaluators/transport/ThermoMechanicalCoefficients_Def.hpp"
  "${LCM_DIR}/evaluators/transport/ThermoMechanicalCoefficients.hpp"
  "${LCM_DIR}/evaluators/transport/TransportCoefficients_Def.hpp"
  "${LCM_DIR}/evaluators/transport/TransportCoefficients.hpp"
  "${LCM_DIR}/evaluators/transport/TransportResidual_Def.hpp"
  "${LCM_DIR}/evaluators/transport/TransportResidual.hpp"
)

#LCM models
set(models-sources
  "${LCM_DIR}/models/AAAModel.cpp"
  "${LCM_DIR}/models/AbstractModel.cpp"
  "${LCM_DIR}/models/AnisotropicDamageModel.cpp"
  "${LCM_DIR}/models/AnisotropicHyperelasticDamageModel.cpp"
  "${LCM_DIR}/models/AnisotropicViscoplasticModel.cpp"
  "${LCM_DIR}/models/CapExplicitModel.cpp"
  "${LCM_DIR}/models/CapImplicitModel.cpp"
  "${LCM_DIR}/models/ConstitutiveModel.cpp"
  "${LCM_DIR}/models/ConstitutiveModelDriver.cpp"
  "${LCM_DIR}/models/ConstitutiveModelDriverPre.cpp"
  "${LCM_DIR}/models/ConstitutiveModelInterface.cpp"
  "${LCM_DIR}/models/ConstitutiveModelParameters.cpp"
  "${LCM_DIR}/models/CreepModel.cpp"
  "${LCM_DIR}/models/DruckerPragerModel.cpp"
  "${LCM_DIR}/models/ElasticCrystalModel.cpp"
  "${LCM_DIR}/models/ElasticDamageModel.cpp"
  "${LCM_DIR}/models/ElastoViscoplasticModel.cpp"
  "${LCM_DIR}/models/HyperelasticDamageModel.cpp"
  "${LCM_DIR}/models/J2FiberModel.cpp"
  "${LCM_DIR}/models/J2HMCModel.cpp"
  "${LCM_DIR}/models/J2MiniSolver.cpp"
  "${LCM_DIR}/models/J2Model.cpp"
  "${LCM_DIR}/models/LinearElasticModel.cpp"
  "${LCM_DIR}/models/LinearHMCModel.cpp"
  "${LCM_DIR}/models/LinearPiezoModel.cpp"
  "${LCM_DIR}/models/MooneyRivlinModel.cpp"
  "${LCM_DIR}/models/NeohookeanModel.cpp"
  "${LCM_DIR}/models/NewtonianFluidModel.cpp"
  "${LCM_DIR}/models/OrtizPandolfiModel.cpp"
  "${LCM_DIR}/models/RIHMRModel.cpp"
  "${LCM_DIR}/models/StVenantKirchhoffModel.cpp"
  "${LCM_DIR}/models/TvergaardHutchinsonModel.cpp"
  "${LCM_DIR}/models/ViscoElasticModel.cpp"
  "${LCM_DIR}/parallel_models/ParallelNeohookeanModel.cpp"
)
set(models-headers
  "${LCM_DIR}/models/AAAModel_Def.hpp"
  "${LCM_DIR}/models/AAAModel.hpp"
  "${LCM_DIR}/models/AbstractModel.hpp"
  "${LCM_DIR}/models/AnisotropicDamageModel_Def.hpp"
  "${LCM_DIR}/models/AnisotropicDamageModel.hpp"
  "${LCM_DIR}/models/AnisotropicHyperelasticDamageModel_Def.hpp"
  "${LCM_DIR}/models/AnisotropicHyperelasticDamageModel.hpp"
  "${LCM_DIR}/models/AnisotropicViscoplasticModel_Def.hpp"
  "${LCM_DIR}/models/AnisotropicViscoplasticModel.hpp"
  "${LCM_DIR}/models/CapExplicitModel_Def.hpp"
  "${LCM_DIR}/models/CapExplicitModel.hpp"
  "${LCM_DIR}/models/CapImplicitModel_Def.hpp"
  "${LCM_DIR}/models/CapImplicitModel.hpp"
  "${LCM_DIR}/models/ConstitutiveModel_Def.hpp"
  "${LCM_DIR}/models/ConstitutiveModelDriver_Def.hpp"
  "${LCM_DIR}/models/ConstitutiveModelDriver.hpp"
  "${LCM_DIR}/models/ConstitutiveModelDriverPre_Def.hpp"
  "${LCM_DIR}/models/ConstitutiveModelDriverPre.hpp"
  "${LCM_DIR}/models/ConstitutiveModel.hpp"
  "${LCM_DIR}/models/ConstitutiveModelInterface_Def.hpp"
  "${LCM_DIR}/models/ConstitutiveModelInterface.hpp"
  "${LCM_DIR}/models/ConstitutiveModelParameters_Def.hpp"
  "${LCM_DIR}/models/ConstitutiveModelParameters.hpp"
  "${LCM_DIR}/models/CreepModel_Def.hpp"
  "${LCM_DIR}/models/CreepModel.hpp"
  "${LCM_DIR}/models/DruckerPragerModel_Def.hpp"
  "${LCM_DIR}/models/DruckerPragerModel.hpp"
  "${LCM_DIR}/models/ElasticCrystalModel_Def.hpp"
  "${LCM_DIR}/models/ElasticCrystalModel.hpp"
  "${LCM_DIR}/models/ElasticDamageModel_Def.hpp"
  "${LCM_DIR}/models/ElasticDamageModel.hpp"
  "${LCM_DIR}/models/ElastoViscoplasticCore_Def.hpp"
  "${LCM_DIR}/models/ElastoViscoplasticCore.hpp"
  "${LCM_DIR}/models/ElastoViscoplasticModel_Def.hpp"
  "${LCM_DIR}/models/ElastoViscoplasticModel.hpp"
  "${LCM_DIR}/models/HyperelasticDamageModel_Def.hpp"
  "${LCM_DIR}/models/HyperelasticDamageModel.hpp"
  "${LCM_DIR}/models/J2FiberModel_Def.hpp"
  "${LCM_DIR}/models/J2FiberModel.hpp"
  "${LCM_DIR}/models/J2HMCModel_Def.hpp"
  "${LCM_DIR}/models/J2HMCModel.hpp"
  "${LCM_DIR}/models/J2MiniSolver_Def.hpp"
  "${LCM_DIR}/models/J2MiniSolver.hpp"
  "${LCM_DIR}/models/J2Model_Def.hpp"
  "${LCM_DIR}/models/J2Model.hpp"
  "${LCM_DIR}/models/LinearElasticModel_Def.hpp"
  "${LCM_DIR}/models/LinearElasticModel.hpp"
  "${LCM_DIR}/models/LinearHMCModel_Def.hpp"
  "${LCM_DIR}/models/LinearHMCModel.hpp"
  "${LCM_DIR}/models/LinearPiezoModel_Def.hpp"
  "${LCM_DIR}/models/LinearPiezoModel.hpp"
  "${LCM_DIR}/models/MooneyRivlinModel_Def.hpp"
  "${LCM_DIR}/models/MooneyRivlinModel.hpp"
  "${LCM_DIR}/models/NeohookeanModel_Def.hpp"
  "${LCM_DIR}/models/NeohookeanModel.hpp"
  "${LCM_DIR}/models/NewtonianFluidModel_Def.hpp"
  "${LCM_DIR}/models/NewtonianFluidModel.hpp"
  "${LCM_DIR}/models/OrtizPandolfiModel_Def.hpp"
  "${LCM_DIR}/models/OrtizPandolfiModel.hpp"
  "${LCM_DIR}/models/RIHMRModel_Def.hpp"
  "${LCM_DIR}/models/RIHMRModel.hpp"
  "${LCM_DIR}/models/StVenantKirchhoffModel_Def.hpp"
  "${LCM_DIR}/models/StVenantKirchhoffModel.hpp"
  "${LCM_DIR}/models/TvergaardHutchinsonModel_Def.hpp"
  "${LCM_DIR}/models/TvergaardHutchinsonModel.hpp"
  "${LCM_DIR}/models/ViscoElasticModel_Def.hpp"
  "${LCM_DIR}/models/ViscoElasticModel.hpp"
  "${LCM_DIR}/models/core/CrystalPlasticity/CrystalPlasticityCore_Def.hpp"
  "${LCM_DIR}/models/core/CrystalPlasticity/CrystalPlasticityCore.hpp"
  "${LCM_DIR}/models/core/CrystalPlasticity/CrystalPlasticityFwd.hpp"
  "${LCM_DIR}/models/core/CrystalPlasticity/FlowRule_Def.hpp"
  "${LCM_DIR}/models/core/CrystalPlasticity/FlowRule.hpp"
  "${LCM_DIR}/models/core/CrystalPlasticity/HardeningLaw_Def.hpp"
  "${LCM_DIR}/models/core/CrystalPlasticity/HardeningLaw.hpp"
  "${LCM_DIR}/models/core/CrystalPlasticity/Integrator_Def.hpp"
  "${LCM_DIR}/models/core/CrystalPlasticity/Integrator.hpp"
  "${LCM_DIR}/models/core/CrystalPlasticity/NonlinearSolver_Def.hpp"
  "${LCM_DIR}/models/core/CrystalPlasticity/NonlinearSolver.hpp"
  "${LCM_DIR}/models/core/CrystalPlasticity/ParameterReader_Def.hpp"
  "${LCM_DIR}/models/core/CrystalPlasticity/ParameterReader.hpp"
  "${LCM_DIR}/parallel_models/ParallelConstitutiveModel_Def.hpp"
  "${LCM_DIR}/parallel_models/ParallelConstitutiveModel.hpp"
  "${LCM_DIR}/parallel_models/ParallelNeohookeanModel_Def.hpp"
  "${LCM_DIR}/parallel_models/ParallelNeohookeanModel.hpp"
)
IF (NOT Kokkos_ENABLE_Cuda)
set(models-sources ${models-sources}
  "${LCM_DIR}/models/GursonHMRModel.cpp"
  "${LCM_DIR}/models/GursonModel.cpp"
  "${LCM_DIR}/models/FM_AlbanyInterface.cpp"
  "${LCM_DIR}/models/FerroicCore.cpp"
  "${LCM_DIR}/models/CrystalPlasticityModel.cpp"
)
set(models-headers ${models-headers}
  "${LCM_DIR}/models/GursonHMRModel_Def.hpp"
  "${LCM_DIR}/models/GursonHMRModel.hpp"
  "${LCM_DIR}/models/GursonModel_Def.hpp"
  "${LCM_DIR}/models/GursonModel.hpp"
  "${LCM_DIR}/models/FM_AlbanyInterface_Def.hpp"
  "${LCM_DIR}/models/FM_AlbanyInterface.hpp"
  "${LCM_DIR}/models/FerroicCore_Def.hpp"
  "${LCM_DIR}/models/FerroicCore.hpp"
  "${LCM_DIR}/models/FerroicModel_Def.hpp"
  "${LCM_DIR}/models/FerroicModel.hpp"
  "${LCM_DIR}/models/CrystalPlasticityModel_Def.hpp"
  "${LCM_DIR}/models/CrystalPlasticityModel.hpp"
)
ENDIF (NOT Kokkos_ENABLE_Cuda)

#LCM utils
set(utils-sources
  "${LCM_DIR}/utils/Projection.cpp"
  "${LCM_DIR}/utils/LocalNonlinearSolver.cpp"
  "${LCM_DIR}/utils/NOXSolverPrePostOperator.cc"
  "${LCM_DIR}/utils/NOX_StatusTest_ModelEvaluatorFlag.cc"
)
set(utils-headers
  "${LCM_DIR}/utils/Projection.hpp"
  "${LCM_DIR}/utils/LocalNonlinearSolver.hpp"
  "${LCM_DIR}/utils/LocalNonlinearSolver_Def.hpp"
  "${LCM_DIR}/utils/NOXSolverPrePostOperator.h"
  "${LCM_DIR}/utils/NOX_StatusTest_ModelEvaluatorFlag.h"
)
IF(ALBANY_HAVE_STK)
  set(utils-sources ${utils-sources}
    "${LCM_DIR}/utils/LCMPartition.cc"
    "${LCM_DIR}/utils/MiniLinearSolver.cc"
    "${LCM_DIR}/utils/MiniNonlinearSolver.cc"
  )
  set(utils-headers ${utils-headers}
    "${LCM_DIR}/utils/LCMPartition.h"
    "${LCM_DIR}/utils/MiniLinearSolver.h"
    "${LCM_DIR}/utils/MiniLinearSolver.t.h"
    "${LCM_DIR}/utils/MiniNonlinearSolver.h"
    "${LCM_DIR}/utils/MiniNonlinearSolver.t.h"
    "${LCM_DIR}/utils/MiniSolvers.h"
  )
  IF (ALBANY_PERIDIGM AND ALBANY_EPETRA)
    set(utils-sources ${utils-sources}
      "${LCM_DIR}/utils/PeridigmManager.cpp"
    )
    set(utils-headers ${utils-headers}
      "${LCM_DIR}/utils/PeridigmManager.hpp"
    )
  ENDIF(ALBANY_PERIDIGM AND ALBANY_EPETRA)
ENDIF(ALBANY_HAVE_STK)

#LCM utils/topology
IF(ALBANY_HAVE_STK AND ALBANY_BGL)
  set(topology-sources
    "${LCM_DIR}/utils/topology/BarycentricSubdivision.cc"
    "${LCM_DIR}/utils/topology/MinimumSurface.cc"
    "${LCM_DIR}/utils/topology/Subgraph.cc"
    "${LCM_DIR}/utils/topology/Topology.cc"
    "${LCM_DIR}/utils/topology/Topology_FractureCriterion.cc"
    "${LCM_DIR}/utils/topology/Topology_Utils.cc"
  )
  set(topology-headers
    "${LCM_DIR}/utils/topology/Subgraph.h"
    "${LCM_DIR}/utils/topology/Topology_FractureCriterion.h"
    "${LCM_DIR}/utils/topology/Topology.h"
    "${LCM_DIR}/utils/topology/Topology_Types.h"
    "${LCM_DIR}/utils/topology/Topology_Utils.h"
  )
ENDIF()

set(SOURCES ${SOURCES}
  ${problem-sources}
  ${evaluator-sources}
  ${model-eval-sources}
  ${surface-element-sources}
  ${transport-sources}
  ${hmc-sources}
  ${pd-sources}
  ${poro-sources}
  ${bc-sources}
  ${models-sources}
  ${utils-sources}
  ${topology-sources}
  ${contact-sources}
  ${residual-sources}
  ${lin-sources}
  ${proj-sources}
  ${response-sources}
  ${kin-sources}
  ${sf-sources}
  )
set(HEADERS ${HEADERS}
  ${problem-headers}
  ${evaluator-headers}
  ${model-eval-headers}
  ${surface-element-headers}
  ${transport-headers}
  ${hmc-headers}
  ${pd-headers}       
  ${poro-headers}
  ${bc-headers}
  ${models-headers}
  ${utils-headers}
  ${topology-headers}
  ${contact-headers}
  ${residual-headers}
  ${lin-headers}
  ${proj-headers}
  ${response-headers}
  ${kin-headers}
  ${sf-headers}
 )

IF(ALBANY_LAME OR ALBANY_LAMENT)
  SET(SOURCES ${SOURCES}
    ${LCM_DIR}/utils/lame/LameUtils.cpp
    ${LCM_DIR}/problems/lame/LameProblem.cpp
    )
  SET(HEADERS ${HEADERS}
    ${LCM_DIR}/utils/lame/LameUtils.hpp
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
  ${LCM_DIR}/evaluators/contact
  ${LCM_DIR}/evaluators/HMC
  ${LCM_DIR}/evaluators/kinematics
  ${LCM_DIR}/evaluators/lame
  ${LCM_DIR}/evaluators/linear-elasticity
  ${LCM_DIR}/evaluators/peridigm
  ${LCM_DIR}/evaluators/poro
  ${LCM_DIR}/evaluators/projection
  ${LCM_DIR}/evaluators/responses
  ${LCM_DIR}/evaluators/residuals
  ${LCM_DIR}/evaluators/surface-element
  ${LCM_DIR}/evaluators/size-field
  ${LCM_DIR}/evaluators/transport
  ${LCM_DIR}/evaluators/HMC
  ${LCM_DIR}/models
  ${LCM_DIR}/utils
  ${LCM_DIR}/utils/topology
  ${LCM_DIR}/utils/dtk_interp_and_error
  ${ALBANY_PUMI_INCLUDE_DIRS}
  ${Trilinos_TPL_INCLUDE_DIRS}
)

#message("include dirs - ${INCLUDE_DIRS}")

include_directories(${INCLUDE_DIRS})
IF(ALBANY_STK_REBALANCE)
  include_directories (${Albany_SOURCE_DIR}/src/disc/stk/percept
     ${Albany_SOURCE_DIR}/src/disc/stk/percept/stk_rebalance
     ${Albany_SOURCE_DIR}/src/disc/stk/percept/stk_rebalance_utils)
endif()

link_directories (${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})

add_library(LCM ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})
set_target_properties(LCM PROPERTIES
  PUBLIC_HEADER "${HEADERS}")

IF (INSTALL_ALBANY)
  install(TARGETS LCM EXPORT albany-export
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()

IF (LCM_TEST_EXES AND ALBANY_DTK) 
  add_executable(DTK_Interp_and_Error utils/dtk_interp_and_error/dtk_interp_and_error.cpp)
  add_executable(DTK_Interp_Volume_to_NS utils/dtk_interp_and_error/interpolation_volume_to_ns.cpp)
ENDIF()

IF (LCM_TEST_EXES AND ALBANY_BGL) 
  add_executable(BifurcationTest test/utils/BifurcationTest.cc)
  add_executable(MaterialPointSimulator test/utils/MaterialPointSimulator.cc)
  add_executable(BoundarySurfaceOutput test/utils/BoundarySurfaceOutput.cc)
  add_executable(MeshComponents test/utils/MeshComponents.cc)
  add_executable(MinSurfaceMPS test/utils/MinSurfaceMPS.cc)
  add_executable(MinSurfaceOutput test/utils/MinSurfaceOutput.cc)
  add_executable(NodeUpdate test/utils/NodeUpdate.cc)
  add_executable(PartitionTest test/utils/PartitionTest.cc)
  add_executable(Subdivision test/utils/Subdivision.cc)
  add_executable(Test1_Subdivision test/utils/Test1_Subdivision.cc)
  add_executable(Test2_Subdivision test/utils/Test2_Subdivision.cc)
  add_executable(TopologyBase test/utils/TopologyBase.cc)

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

  add_executable(
    utHeliumODEs
    test/unit_tests/StandardUnitTestMain.cpp
    test/unit_tests/utHeliumODEs.cpp
    )

  IF(NOT BUILD_SHARED_LIBS)
    add_executable(utStaticAllocator test/unit_tests/utStaticAllocator.cpp)
  ENDIF()

  add_executable(utMiniSolvers test/unit_tests/utMiniSolvers.cc)

  IF (ALBANY_ROL)
    add_executable(utMiniSolversROL test/unit_tests/utMiniSolversROL.cc)
  ENDIF()

ENDIF()

IF (LCM_TEST_EXES AND (ALBANY_LAME OR ALBANY_LAMENT))
  IF(ALBANY_64BIT_INT)
    IF(ALBANY_EPETRA) 
      add_executable(utLameStress_elastic
        ${Albany_SOURCE_DIR}/src/PHAL_Dimension.cpp
        ${Albany_SOURCE_DIR}/src/PHAL_AlbanyTraits.cpp
        ${Albany_SOURCE_DIR}/src/Albany_Utils.cpp
    	${Albany_SOURCE_DIR}/src/Albany_StateManager.cpp
    	${Albany_SOURCE_DIR}/src/Petra_Converters_64.cpp
    	${Albany_SOURCE_DIR}/src/disc/stk/Albany_GenericSTKMeshStruct.cpp
    	${Albany_SOURCE_DIR}/src/disc/stk/Albany_GenericSTKFieldContainer.cpp
    	${Albany_SOURCE_DIR}/src/disc/stk/Albany_MultiSTKFieldContainer.cpp
    	${Albany_SOURCE_DIR}/src/disc/stk/Albany_OrdinarySTKFieldContainer.cpp
    	${Albany_SOURCE_DIR}/src/disc/stk/Albany_TmplSTKMeshStruct.cpp
    	${Albany_SOURCE_DIR}/src/disc/stk/Albany_STKNodeSharing.cpp
    	${Albany_SOURCE_DIR}/src/disc/stk/Albany_STKDiscretization.cpp
    	${Albany_SOURCE_DIR}/src/disc/stk/Albany_STKNodeFieldContainer.cpp
    	${Albany_SOURCE_DIR}/src/LCM/utils/MaterialDatabase.cc
    	utils/lame/LameUtils.cpp
    	evaluators/lame/LameStress.cpp
    	evaluators/SetField.cpp
    	test/unit_tests/StandardUnitTestMain.cpp
    	test/unit_tests/utLameStress_elastic.cpp
    	)
      ELSE()
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
    	  ${Albany_SOURCE_DIR}/src/disc/stk/Albany_STKNodeSharing.cpp
    	  ${Albany_SOURCE_DIR}/src/disc/stk/Albany_STKDiscretization.cpp
    	  ${Albany_SOURCE_DIR}/src/disc/stk/Albany_STKNodeFieldContainer.cpp
    	  ${Albany_SOURCE_DIR}/src/LCM/utils/MaterialDatabase.cc
    	  utils/lame/LameUtils.cpp
    	  evaluators/lame/LameStress.cpp
    	  evaluators/SetField.cpp
    	  test/unit_tests/StandardUnitTestMain.cpp
    	  test/unit_tests/utLameStress_elastic.cpp
   	 )
       ENDIF()
  ELSE()
    IF(ALBANY_EPETRA) 
    	add_executable(utLameStress_elastic
      	  ${Albany_SOURCE_DIR}/src/PHAL_Dimension.cpp
    	  ${Albany_SOURCE_DIR}/src/PHAL_AlbanyTraits.cpp
    	  ${Albany_SOURCE_DIR}/src/Albany_Utils.cpp
    	  ${Albany_SOURCE_DIR}/src/Albany_StateManager.cpp
    	  ${Albany_SOURCE_DIR}/src/Petra_Converters.cpp
    	  ${Albany_SOURCE_DIR}/src/disc/stk/Albany_GenericSTKMeshStruct.cpp
    	  ${Albany_SOURCE_DIR}/src/disc/stk/Albany_GenericSTKFieldContainer.cpp
    	  ${Albany_SOURCE_DIR}/src/disc/stk/Albany_MultiSTKFieldContainer.cpp
    	  ${Albany_SOURCE_DIR}/src/disc/stk/Albany_OrdinarySTKFieldContainer.cpp
    	  ${Albany_SOURCE_DIR}/src/disc/stk/Albany_TmplSTKMeshStruct.cpp
    	  ${Albany_SOURCE_DIR}/src/disc/stk/Albany_STKNodeSharing.cpp
    	  ${Albany_SOURCE_DIR}/src/disc/stk/Albany_STKDiscretization.cpp
    	  ${Albany_SOURCE_DIR}/src/disc/stk/Albany_STKNodeFieldContainer.cpp
    	  ${Albany_SOURCE_DIR}/src/LCM/utils/MaterialDatabase.cc
    	  utils/lame/LameUtils.cpp
    	  evaluators/lame/LameStress.cpp
    	  evaluators/SetField.cpp
    	  test/unit_tests/StandardUnitTestMain.cpp
    	  test/unit_tests/utLameStress_elastic.cpp
         )
    ELSE() 
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
    	  ${Albany_SOURCE_DIR}/src/disc/stk/Albany_STKNodeSharing.cpp
    	  ${Albany_SOURCE_DIR}/src/disc/stk/Albany_STKDiscretization.cpp
    	  ${Albany_SOURCE_DIR}/src/disc/stk/Albany_STKNodeFieldContainer.cpp
    	  ${Albany_SOURCE_DIR}/src/LCM/utils/MaterialDatabase.cc
    	  utils/lame/LameUtils.cpp
    	  evaluators/lame/LameStress.cpp
    	  evaluators/SetField.cpp
    	  test/unit_tests/StandardUnitTestMain.cpp
    	  test/unit_tests/utLameStress_elastic.cpp
         )
      ENDIF()
  ENDIF()

  IF(ALBANY_STK_REBALANCE)
    target_link_libraries(utLameStress_elastic albanySTKRebalance albanyLib ${LAME_LIB} ${LAMENT_LIB} ${MATH_TOOLKIT_LIB} ${ALB_TRILINOS_LIBS} ${Trilinos_EXTRA_LD_FLAGS})
  ELSE ()
    target_link_libraries(utLameStress_elastic albanyLib ${LAME_LIB} ${LAMENT_LIB} ${MATH_TOOLKIT_LIB} ${ALB_TRILINOS_LIBS} ${Trilinos_EXTRA_LD_FLAGS})
  ENDIF ()
ENDIF ()

IF(BUILD_SHARED_LIBS)
  IF(ALBANY_SCOREC)
    SET(LCM_UT_LIBS albanyLib albanyAdapt albanyPUMI albanySTK LCM albanyLib)
  ELSE()
    SET(LCM_UT_LIBS albanyLib albanyAdapt albanySTK LCM albanyLib)
  ENDIF()
ELSE(BUILD_SHARED_LIBS)
  # We really don't link to anything in Hydride
  IF(ALBANY_SCOREC)
    SET(LCM_UT_LIBS LCM albanyAdapt albanyLib albanyPUMI albanySTK albanyLib)
  ELSE()
    SET(LCM_UT_LIBS LCM albanyAdapt albanyLib albanySTK albanyLib )
  ENDIF()
ENDIF(BUILD_SHARED_LIBS)

IF(ALBANY_STK_REBALANCE)
  SET(LCM_UT_LIBS ${LCM_UT_LIBS} albanySTKRebalance)
ENDIF()
   
#Linking tests
IF (LCM_TEST_EXES AND ALBANY_DTK)
  set (repeat_libs ${LCM_UT_LIBS} ${ALBANY_LIBRARIES} ${LCM_UT_LIBS} ${ALBANY_LIBRARIES})
  target_link_libraries(DTK_Interp_and_Error ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(DTK_Interp_Volume_to_NS ${repeat_libs} ${ALL_LIBRARIES})
ENDIF()
IF (LCM_TEST_EXES AND ALBANY_BGL)
  set (repeat_libs ${LCM_UT_LIBS} ${ALBANY_LIBRARIES} ${LCM_UT_LIBS} ${ALBANY_LIBRARIES})
  target_link_libraries(BifurcationTest ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(BoundarySurfaceOutput ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(MaterialPointSimulator ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(MeshComponents ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(MinSurfaceMPS ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(MinSurfaceOutput ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(NodeUpdate ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(PartitionTest ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(Subdivision ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(Test1_Subdivision ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(Test2_Subdivision ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(TopologyBase ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(utLocalNonlinearSolver ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(utMiniSolvers ${ALL_LIBRARIES})
  IF (ALBANY_ROL)
    target_link_libraries(utMiniSolversROL ${ALL_LIBRARIES})
  ENDIF()
  target_link_libraries(utSurfaceElement ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(utHeliumODEs ${repeat_libs} ${ALL_LIBRARIES})
  IF(NOT BUILD_SHARED_LIBS)
    target_link_libraries(utStaticAllocator ${repeat_libs} ${ALL_LIBRARIES})
  ENDIF()
ENDIF()
