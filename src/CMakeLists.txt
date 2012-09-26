#/********************************************************************\
#*            Albany, Copyright (2010) Sandia Corporation             *
#*                                                                    *
#* Notice: This computer software was prepared by Sandia Corporation, *
#* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
#* the Department of Energy (DOE). All rights in the computer software*
#* are reserved by DOE on behalf of the United States Government and  *
#* the Contractor as provided in the Contract. You are authorized to  *
#* use this computer software for Governmental purposes but it is not *
#* to be released or distributed to the public. NEITHER THE GOVERNMENT*
#* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
#* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
#* including this sentence must appear on any copies of this software.*
#*    Questions to Andy Salinger, agsalin@sandia.gov                  *
#\********************************************************************/

#
SET(SOURCES
        Albany_SolverFactory.cpp
        Albany_Utils.cpp
        PHAL_Dimension.cpp
        PHAL_AlbanyTraits.cpp
        Albany_Application.cpp
        Albany_Adaptation.cpp
        Albany_ObserverFactory.cpp
        Albany_RythmosObserver.cpp
        Albany_NOXObserver.cpp
        Albany_StateManager.cpp
        Albany_ModelFactory.cpp
        Albany_ModelEvaluator.cpp
        Albany_SaveEigenData.cpp
        Albany_ExodusOutput.cpp
  )

SET(HEADERS
        Albany_Adaptation.hpp
        Albany_SolverFactory.hpp
        Albany_Utils.hpp
        Albany_DataTypes.hpp
        PHAL_Dimension.hpp
        PHAL_FactoryTraits.hpp
        PHAL_AlbanyTraits.hpp
        PHAL_Workset.hpp
        PHAL_TypeKeyMap.hpp
        Albany_Application.hpp
        Albany_ObserverFactory.hpp
        Albany_RythmosObserver.hpp
        Albany_NOXObserver.hpp
        Albany_StateManager.hpp
        Albany_StateInfoStruct.hpp
        Albany_EigendataInfoStruct.hpp
        Albany_ModelFactory.hpp
        Albany_ModelEvaluator.hpp
        Albany_SaveEigenData.hpp
        Albany_ExodusOutput.hpp
        Albany_DummyParameterAccessor.hpp
  )

#problems
SET(SOURCES ${SOURCES}
        problems/Albany_AbstractProblem.cpp
        problems/Albany_CahnHillProblem.cpp
        problems/Albany_ProblemUtils.cpp
        problems/Albany_Layouts.cpp
        problems/Albany_EvaluatorUtils.cpp
        problems/Albany_ResponseUtilities.cpp
        problems/Albany_BCUtils.cpp
        problems/Albany_ProblemFactory.cpp
        problems/Albany_HeatProblem.cpp
        problems/Albany_MultiHeatProblem.cpp
        problems/Albany_NavierStokes.cpp
        problems/Albany_GPAMProblem.cpp
        problems/Albany_LinComprNSProblem.cpp
        problems/Albany_ODEProblem.cpp
        problems/Albany_Helmholtz2DProblem.cpp
        problems/Albany_ThermoElectrostaticsProblem.cpp
        problems/Albany_AnalyticFunction.cpp
        problems/Albany_InitialCondition.cpp
  )
SET(HEADERS ${HEADERS}
        problems/Albany_ProblemFactory.hpp
        problems/Albany_AbstractProblem.hpp
        problems/Albany_CahnHillProblem.hpp
        problems/Albany_ProblemUtils.hpp
        problems/Albany_Layouts.hpp
        problems/Albany_EvaluatorUtils.hpp
        problems/Albany_EvaluatorUtils_Def.hpp
        problems/Albany_ResponseUtilities.hpp
        problems/Albany_ResponseUtilities_Def.hpp
        problems/Albany_BCUtils.hpp
        problems/Albany_BCUtils_Def.hpp
        problems/Albany_HeatProblem.hpp
        problems/Albany_MultiHeatProblem.hpp
        problems/Albany_NavierStokes.hpp
        problems/Albany_GPAMProblem.hpp
        problems/Albany_LinComprNSProblem.hpp
        problems/Albany_ODEProblem.hpp
        problems/Albany_Helmholtz2DProblem.hpp
        problems/Albany_ThermoElectrostaticsProblem.hpp
        problems/Albany_AnalyticFunction.hpp
        problems/Albany_InitialCondition.hpp
  )

#responses
SET(SOURCES ${SOURCES}
        responses/Albany_ScalarResponseFunction.cpp
        responses/Albany_AggregateScalarResponseFunction.cpp
        responses/Albany_SamplingBasedScalarResponseFunction.cpp
        responses/Albany_SolutionAverageResponseFunction.cpp
        responses/Albany_SolutionMaxValueResponseFunction.cpp
        responses/Albany_SolutionTwoNormResponseFunction.cpp
        responses/Albany_SolutionFileResponseFunction.cpp
        responses/Albany_FieldManagerScalarResponseFunction.cpp
        responses/Albany_DistributedResponseFunction.cpp
        responses/Albany_SolutionResponseFunction.cpp
        responses/Albany_KLResponseFunction.cpp
        responses/Albany_ResponseFactory.cpp
  )
SET(HEADERS ${HEADERS}
        responses/Albany_AbstractResponseFunction.hpp
        responses/Albany_ScalarResponseFunction.hpp
        responses/Albany_AggregateScalarResponseFunction.hpp
        responses/Albany_SamplingBasedScalarResponseFunction.hpp
        responses/Albany_SolutionAverageResponseFunction.hpp
        responses/Albany_SolutionMaxValueResponseFunction.hpp
        responses/Albany_SolutionTwoNormResponseFunction.hpp
        responses/Albany_SolutionFileResponseFunction.hpp
        responses/Albany_FieldManagerScalarResponseFunction.hpp
        responses/Albany_DistributedResponseFunction.hpp
        responses/Albany_SolutionResponseFunction.hpp
        responses/Albany_KLResponseFunction.hpp
        responses/Albany_ResponseFactory.hpp
  )

#stk
SET(SOURCES ${SOURCES}
        stk/Albany_STKDiscretization.cpp
        stk/Albany_DiscretizationFactory.cpp
        stk/Albany_GenericSTKMeshStruct.cpp
        stk/Albany_TmplSTKMeshStruct.cpp
        stk/Albany_FromCubitSTKMeshStruct.cpp
        stk/Albany_IossSTKMeshStruct.cpp
  )
SET(HEADERS ${HEADERS}
        stk/Albany_AbstractDiscretization.hpp
        stk/Albany_STKDiscretization.hpp
        stk/Albany_DiscretizationFactory.hpp
        stk/Albany_AbstractSTKMeshStruct.hpp
        stk/Albany_GenericSTKMeshStruct.hpp
        stk/Albany_TmplSTKMeshStruct.hpp
        stk/Albany_FromCubitSTKMeshStruct.hpp
        stk/Albany_IossSTKMeshStruct.hpp
  )

#evaluators
SET(SOURCES ${SOURCES}
        evaluators/PHAL_GatherSolution.cpp
        evaluators/PHAL_GatherEigenvectors.cpp
        evaluators/PHAL_ScatterResidual.cpp
        evaluators/PHAL_Dirichlet.cpp
        evaluators/PHAL_Neumann.cpp
        evaluators/PHAL_Source.cpp
        evaluators/PHAL_Constant.cpp
        evaluators/PHAL_ODEResid.cpp
        evaluators/PHAL_ThermalConductivity.cpp
        evaluators/PHAL_Absorption.cpp
        evaluators/PHAL_HeatEqResid.cpp
        evaluators/PHAL_GPAMResid.cpp
        evaluators/PHAL_LinComprNSResid.cpp
        evaluators/PHAL_ComputeBasisFunctions.cpp
        evaluators/PHAL_DOFInterpolation.cpp
        evaluators/PHAL_DOFGradInterpolation.cpp
        evaluators/PHAL_DOFVecInterpolation.cpp
        evaluators/PHAL_DOFVecGradInterpolation.cpp
        evaluators/PHAL_MapToPhysicalFrame.cpp
        evaluators/PHAL_HelmholtzResid.cpp
        evaluators/PHAL_GatherCoordinateVector.cpp
        evaluators/PHAL_CahnHillRhoResid.cpp
        evaluators/PHAL_CahnHillWResid.cpp
        evaluators/PHAL_CahnHillElasResid.cpp
        evaluators/PHAL_CahnHillChemTerm.cpp
        evaluators/PHAL_LangevinNoiseTerm.cpp
        evaluators/PHAL_JouleHeating.cpp
        evaluators/PHAL_TEProp.cpp
        evaluators/PHAL_SaveStateField.cpp
        evaluators/PHAL_LoadStateField.cpp
        evaluators/PHAL_SharedParameter.cpp
        evaluators/PHAL_NSContinuityResid.cpp
        evaluators/PHAL_NSMomentumResid.cpp
        evaluators/PHAL_NSThermalEqResid.cpp
        evaluators/PHAL_NSNeutronEqResid.cpp
        evaluators/PHAL_NSRm.cpp
        evaluators/PHAL_NSContravarientMetricTensor.cpp
        evaluators/PHAL_NSTauM.cpp
        evaluators/PHAL_NSTauT.cpp
        evaluators/PHAL_NSMaterialProperty.cpp
        evaluators/PHAL_NSBodyForce.cpp
        evaluators/PHAL_NSPermeabilityTerm.cpp
        evaluators/PHAL_NSForchheimerTerm.cpp
        evaluators/PHAL_ResponseFieldIntegral.cpp
        evaluators/PHAL_ScatterScalarResponse.cpp
        evaluators/PHAL_SeparableScatterScalarResponse.cpp
  )
SET(HEADERS ${HEADERS}
        evaluators/PHAL_GatherSolution.hpp
        evaluators/PHAL_GatherSolution_Def.hpp
        evaluators/PHAL_GatherEigenvectors.hpp
        evaluators/PHAL_GatherEigenvectors_Def.hpp
        evaluators/PHAL_ScatterResidual.hpp
        evaluators/PHAL_ScatterResidual_Def.hpp
        evaluators/PHAL_Dirichlet.hpp
        evaluators/PHAL_Dirichlet_Def.hpp
        evaluators/PHAL_Neumann.hpp
        evaluators/PHAL_Neumann_Def.hpp
        evaluators/PHAL_Source.hpp
        evaluators/PHAL_Source_Def.hpp
        evaluators/PHAL_Constant.hpp
        evaluators/PHAL_Constant_Def.hpp
        evaluators/PHAL_ODEResid.hpp
        evaluators/PHAL_ODEResid_Def.hpp
        evaluators/PHAL_ThermalConductivity.hpp
        evaluators/PHAL_ThermalConductivity_Def.hpp
        evaluators/PHAL_Absorption.hpp
        evaluators/PHAL_Absorption_Def.hpp
        evaluators/PHAL_HeatEqResid.hpp
        evaluators/PHAL_HeatEqResid_Def.hpp
        evaluators/PHAL_GPAMResid.hpp
        evaluators/PHAL_GPAMResid_Def.hpp
        evaluators/PHAL_LinComprNSResid.hpp
        evaluators/PHAL_LinComprNSResid_Def.hpp
        evaluators/PHAL_ComputeBasisFunctions.hpp
        evaluators/PHAL_ComputeBasisFunctions_Def.hpp
        evaluators/PHAL_DOFInterpolation.hpp
        evaluators/PHAL_DOFInterpolation_Def.hpp
        evaluators/PHAL_DOFGradInterpolation.hpp
        evaluators/PHAL_DOFGradInterpolation_Def.hpp
        evaluators/PHAL_DOFVecInterpolation.hpp
        evaluators/PHAL_DOFVecInterpolation_Def.hpp
        evaluators/PHAL_DOFVecGradInterpolation.hpp
        evaluators/PHAL_DOFVecGradInterpolation_Def.hpp
        evaluators/PHAL_MapToPhysicalFrame.hpp
        evaluators/PHAL_MapToPhysicalFrame_Def.hpp
        evaluators/PHAL_HelmholtzResid.hpp
        evaluators/PHAL_HelmholtzResid_Def.hpp
        evaluators/PHAL_GatherCoordinateVector.hpp
        evaluators/PHAL_GatherCoordinateVector_Def.hpp
        evaluators/PHAL_CahnHillRhoResid.hpp
        evaluators/PHAL_CahnHillRhoResid_Def.hpp
        evaluators/PHAL_CahnHillWResid.hpp
        evaluators/PHAL_CahnHillWResid_Def.hpp
        evaluators/PHAL_CahnHillElasResid.hpp
        evaluators/PHAL_CahnHillElasResid_Def.hpp
        evaluators/PHAL_CahnHillChemTerm.hpp
        evaluators/PHAL_CahnHillChemTerm_Def.hpp
        evaluators/PHAL_LangevinNoiseTerm.hpp
        evaluators/PHAL_LangevinNoiseTerm_Def.hpp
        evaluators/PHAL_JouleHeating.hpp
        evaluators/PHAL_JouleHeating_Def.hpp
        evaluators/PHAL_TEProp.hpp
        evaluators/PHAL_TEProp_Def.hpp
        evaluators/PHAL_SaveStateField.hpp
        evaluators/PHAL_SaveStateField_Def.hpp
        evaluators/PHAL_LoadStateField.hpp
        evaluators/PHAL_LoadStateField_Def.hpp
        evaluators/PHAL_SharedParameter.hpp
        evaluators/PHAL_SharedParameter_Def.hpp
        evaluators/PHAL_NSContinuityResid.hpp
        evaluators/PHAL_NSContinuityResid_Def.hpp
        evaluators/PHAL_NSMomentumResid.hpp
        evaluators/PHAL_NSMomentumResid_Def.hpp
        evaluators/PHAL_NSThermalEqResid.hpp
        evaluators/PHAL_NSThermalEqResid_Def.hpp
        evaluators/PHAL_NSNeutronEqResid.hpp
        evaluators/PHAL_NSNeutronEqResid_Def.hpp
        evaluators/PHAL_NSRm.hpp
        evaluators/PHAL_NSRm_Def.hpp
        evaluators/PHAL_NSContravarientMetricTensor.hpp
        evaluators/PHAL_NSContravarientMetricTensor_Def.hpp
        evaluators/PHAL_NSTauM.hpp
        evaluators/PHAL_NSTauM_Def.hpp
        evaluators/PHAL_NSTauT.hpp
        evaluators/PHAL_NSTauT_Def.hpp
        evaluators/PHAL_NSMaterialProperty.hpp
        evaluators/PHAL_NSMaterialProperty_Def.hpp
        evaluators/PHAL_NSBodyForce.hpp
        evaluators/PHAL_NSBodyForce_Def.hpp
        evaluators/PHAL_NSPermeabilityTerm.hpp
        evaluators/PHAL_NSPermeabilityTerm_Def.hpp
        evaluators/PHAL_NSForchheimerTerm.hpp
        evaluators/PHAL_NSForchheimerTerm_Def.hpp
        evaluators/PHAL_ResponseFieldIntegral.hpp
        evaluators/PHAL_ResponseFieldIntegral_Def.hpp
        evaluators/PHAL_ScatterScalarResponse.hpp
        evaluators/PHAL_ScatterScalarResponse_Def.hpp
        evaluators/PHAL_SeparableScatterScalarResponse.hpp
        evaluators/PHAL_SeparableScatterScalarResponse_Def.hpp
  )

# ASCR
IF(ALBANY_ASCR)
  SET(SOURCES ${SOURCES}
    Albany_Networks.cpp
    )
  SET(HEADERS ${HEADERS} 
    Albany_Networks.hpp
    )
ENDIF()

# MOR (Model Order Reduction)
SET(MOR_DIR ${Albany_SOURCE_DIR}/src/MOR)

SET(SOURCES ${SOURCES}
  ${MOR_DIR}/Albany_EpetraSamplingOperator.cpp
  ${MOR_DIR}/Albany_GaussNewtonOperatorFactory.cpp
  ${MOR_DIR}/Albany_PetrovGalerkinOperatorFactory.cpp
  ${MOR_DIR}/Albany_ReducedJacobianFactory.cpp
  ${MOR_DIR}/Albany_ReducedLinearOperatorFactory.cpp
  ${MOR_DIR}/Albany_ReducedSpace.cpp
  ${MOR_DIR}/Albany_GalerkinProjectionOperator.cpp
  ${MOR_DIR}/Albany_ReducedOrderModelEvaluator.cpp
  ${MOR_DIR}/Albany_ReducedOrderModelFactory.cpp
  ${MOR_DIR}/Albany_MORObserverFactory.cpp
  ${MOR_DIR}/Albany_FullStateReconstructor.cpp
  ${MOR_DIR}/Albany_RythmosFullStateReconstructor.cpp
  ${MOR_DIR}/Albany_RythmosStepperFullStateWrapper.cpp
  ${MOR_DIR}/Albany_ProjectionError.cpp
  ${MOR_DIR}/Albany_ProjectionErrorObserver.cpp
  ${MOR_DIR}/Albany_RythmosProjectionErrorObserver.cpp
  ${MOR_DIR}/Albany_SnapshotCollection.cpp
  ${MOR_DIR}/Albany_SnapshotCollectionObserver.cpp
  ${MOR_DIR}/Albany_RythmosSnapshotCollectionObserver.cpp
  ${MOR_DIR}/Albany_BasisInputFile.cpp
  ${MOR_DIR}/Albany_MultiVectorInputFile.cpp
  ${MOR_DIR}/Albany_MultiVectorInputFileFactory.cpp
  ${MOR_DIR}/Albany_MatrixMarketMVInputFile.cpp
  ${MOR_DIR}/Albany_Hdf5MVInputFile.cpp
  ${MOR_DIR}/Albany_MultiVectorOutputFileFactory.cpp
  ${MOR_DIR}/Albany_MatrixMarketMVOutputFile.cpp
  ${MOR_DIR}/Albany_Hdf5MVOutputFile.cpp
)
SET(HEADERS ${HEADERS}
  ${MOR_DIR}/Albany_MORUtils.hpp
  ${MOR_DIR}/Albany_BasisOps.hpp
  ${MOR_DIR}/Albany_EpetraSamplingOperator.hpp
  ${MOR_DIR}/Albany_ReducedOperatorFactory.hpp
  ${MOR_DIR}/Albany_GaussNewtonOperatorFactory.hpp
  ${MOR_DIR}/Albany_PetrovGalerkinOperatorFactory.hpp
  ${MOR_DIR}/Albany_ReducedJacobianFactory.hpp
  ${MOR_DIR}/Albany_ReducedLinearOperatorFactory.hpp
  ${MOR_DIR}/Albany_ReducedSpace.hpp
  ${MOR_DIR}/Albany_GalerkinProjectionOperator.hpp
  ${MOR_DIR}/Albany_ReducedOrderModelEvaluator.hpp
  ${MOR_DIR}/Albany_ReducedOrderModelFactory.hpp
  ${MOR_DIR}/Albany_MORObserverFactory.hpp
  ${MOR_DIR}/Albany_FullStateReconstructor.hpp
  ${MOR_DIR}/Albany_RythmosFullStateReconstructor.hpp
  ${MOR_DIR}/Albany_RythmosStepperFullStateWrapper.hpp
  ${MOR_DIR}/Albany_ProjectionError.hpp
  ${MOR_DIR}/Albany_ProjectionErrorObserver.hpp
  ${MOR_DIR}/Albany_RythmosProjectionErrorObserver.hpp
  ${MOR_DIR}/Albany_SnapshotCollection.hpp
  ${MOR_DIR}/Albany_SnapshotCollectionObserver.hpp
  ${MOR_DIR}/Albany_RythmosSnapshotCollectionObserver.hpp
  ${MOR_DIR}/Albany_RythmosUtils.hpp
  ${MOR_DIR}/Albany_BasisInputFile.hpp
  ${MOR_DIR}/Albany_MultiVectorInputFile.hpp
  ${MOR_DIR}/Albany_MultiVectorInputFileFactory.hpp
  ${MOR_DIR}/Albany_MatrixMarketMVInputFile.hpp
  ${MOR_DIR}/Albany_Hdf5MVInputFile.hpp
  ${MOR_DIR}/Albany_MultiVectorOutputFileFactory.hpp
  ${MOR_DIR}/Albany_MultiVectorOutputFile.hpp
  ${MOR_DIR}/Albany_MatrixMarketMVOutputFile.hpp
  ${MOR_DIR}/Albany_Hdf5MVOutputFile.hpp
)

# LCM
IF(ALBANY_LCM)
  SET(LCM_DIR ${Albany_SOURCE_DIR}/src/LCM)

  # LCM problems
  SET(SOURCES ${SOURCES}
    ${LCM_DIR}/problems/ElasticityProblem.cpp
    ${LCM_DIR}/problems/NonlinearElasticityProblem.cpp
    ${LCM_DIR}/problems/MechanicsProblem.cpp
    ${LCM_DIR}/problems/ThermoElasticityProblem.cpp
    ${LCM_DIR}/problems/GradientDamageProblem.cpp
    ${LCM_DIR}/problems/PoroElasticityProblem.cpp
    ${LCM_DIR}/problems/UnSatPoroElasticityProblem.cpp
    ${LCM_DIR}/problems/TLPoroPlasticityProblem.cpp
    ${LCM_DIR}/problems/ThermoPoroPlasticityProblem.cpp
    ${LCM_DIR}/problems/ThermoMechanicalProblem.cpp
    ${LCM_DIR}/problems/HDiffusionDeformationProblem.cpp
    ${LCM_DIR}/problems/ProjectionProblem.cpp
  )
  IF(ALBANY_MPI)
    SET(SOURCES ${SOURCES}
      ${LCM_DIR}/problems/MesoScaleLinkProblem.cpp
    )
  ENDIF()

  SET(HEADERS ${HEADERS}
    ${LCM_DIR}/problems/ElasticityProblem.hpp
    ${LCM_DIR}/problems/NonlinearElasticityProblem.hpp
    ${LCM_DIR}/problems/MechanicsProblem.hpp
    ${LCM_DIR}/problems/ThermoElasticityProblem.hpp
    ${LCM_DIR}/problems/GradientDamageProblem.hpp
    ${LCM_DIR}/problems/PoroElasticityProblem.hpp
    ${LCM_DIR}/problems/UnSatPoroElasticityProblem.hpp
    ${LCM_DIR}/problems/TLPoroPlasticityProblem.hpp
    ${LCM_DIR}/problems/ThermoPoroPlasticityProblem.hpp
    ${LCM_DIR}/problems/ThermoMechanicalProblem.hpp
    ${LCM_DIR}/problems/HDiffusionDeformationProblem.hpp
    ${LCM_DIR}/problems/ProjectionProblem.hpp
  )
  IF(ALBANY_MPI)
    SET(HEADERS ${HEADERS}
      ${LCM_DIR}/problems/MesoScaleLinkProblem.hpp
    )
  ENDIF()

  # LCM evaluators
  SET(SOURCES ${SOURCES}
    ${LCM_DIR}/evaluators/Stress.cpp
    ${LCM_DIR}/evaluators/CapModelStress.cpp
    ${LCM_DIR}/evaluators/CapImplicit.cpp
    ${LCM_DIR}/evaluators/GursonSDStress.cpp
    ${LCM_DIR}/evaluators/GursonFD.cpp
    ${LCM_DIR}/evaluators/J2Fiber.cpp
    ${LCM_DIR}/evaluators/Strain.cpp
    ${LCM_DIR}/evaluators/ElasticModulus.cpp
    ${LCM_DIR}/evaluators/PoissonsRatio.cpp
    ${LCM_DIR}/evaluators/DefGrad.cpp
    ${LCM_DIR}/evaluators/Neohookean.cpp
    ${LCM_DIR}/evaluators/J2Stress.cpp
    ${LCM_DIR}/evaluators/ElasticityResid.cpp
    ${LCM_DIR}/evaluators/TLElasResid.cpp
    ${LCM_DIR}/evaluators/EnergyPotential.cpp
    ${LCM_DIR}/evaluators/HardeningModulus.cpp
    ${LCM_DIR}/evaluators/YieldStrength.cpp
    ${LCM_DIR}/evaluators/PisdWdF.cpp
    ${LCM_DIR}/evaluators/DamageResid.cpp
    ${LCM_DIR}/evaluators/J2Damage.cpp
    ${LCM_DIR}/evaluators/DamageLS.cpp
    ${LCM_DIR}/evaluators/SaturationModulus.cpp
    ${LCM_DIR}/evaluators/SaturationExponent.cpp
    ${LCM_DIR}/evaluators/Localization.cpp
    ${LCM_DIR}/evaluators/SurfaceBasis.cpp
    ${LCM_DIR}/evaluators/SurfaceVectorJump.cpp
    ${LCM_DIR}/evaluators/SurfaceScalarJump.cpp
    ${LCM_DIR}/evaluators/CurrentCoords.cpp
    ${LCM_DIR}/evaluators/DamageSource.cpp
    ${LCM_DIR}/evaluators/BulkModulus.cpp
    ${LCM_DIR}/evaluators/ShearModulus.cpp
    ${LCM_DIR}/evaluators/KfieldBC.cpp
    ${LCM_DIR}/evaluators/TorsionBC.cpp
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
    ${LCM_DIR}/evaluators/TimeDepBC.cpp
    ${LCM_DIR}/evaluators/DiffusionCoefficient.cpp
    ${LCM_DIR}/evaluators/EffectiveDiffusivity.cpp
    ${LCM_DIR}/evaluators/EquilibriumConstant.cpp
    ${LCM_DIR}/evaluators/TrappedSolvent.cpp
    ${LCM_DIR}/evaluators/TrappedConcentration.cpp
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
    ${LCM_DIR}/evaluators/QptLocation.cpp
    ${LCM_DIR}/evaluators/MooneyRivlin.cpp
    ${LCM_DIR}/evaluators/MooneyRivlinDamage.cpp
    ${LCM_DIR}/evaluators/MooneyRivlin_Incompressible.cpp
    ${LCM_DIR}/evaluators/MooneyRivlin_Incompressible_Damage.cpp
    ${LCM_DIR}/evaluators/RIHMR.cpp
    ${LCM_DIR}/evaluators/GursonHMR.cpp
    ${LCM_DIR}/evaluators/RecoveryModulus.cpp
    ${LCM_DIR}/evaluators/FaceFractureCriteria.cpp
    ${LCM_DIR}/evaluators/FaceAverage.cpp
    ${LCM_DIR}/evaluators/AAA.cpp
    ${LCM_DIR}/evaluators/LatticeDefGrad.cpp
  )

  IF(ALBANY_MPI)
    SET(SOURCES ${SOURCES}
      ${LCM_DIR}/evaluators/MultiScaleStress.cpp
    )
  ENDIF()

  SET(HEADERS ${HEADERS}
    ${LCM_DIR}/evaluators/Stress.hpp
    ${LCM_DIR}/evaluators/Stress_Def.hpp
    ${LCM_DIR}/evaluators/CapModelStress.hpp
    ${LCM_DIR}/evaluators/CapModelStress_Def.hpp
    ${LCM_DIR}/evaluators/CapImplicit.hpp
    ${LCM_DIR}/evaluators/CapImplicit_Def.hpp
    ${LCM_DIR}/evaluators/GursonSDStress.hpp
    ${LCM_DIR}/evaluators/GursonSDStress_Def.hpp
    ${LCM_DIR}/evaluators/GursonFD.hpp
    ${LCM_DIR}/evaluators/GursonFD_Def.hpp
    ${LCM_DIR}/evaluators/J2Fiber.hpp
    ${LCM_DIR}/evaluators/J2Fiber_Def.hpp
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
    ${LCM_DIR}/evaluators/J2Stress.hpp
    ${LCM_DIR}/evaluators/J2Stress_Def.hpp
    ${LCM_DIR}/evaluators/ElasticityResid.hpp
    ${LCM_DIR}/evaluators/ElasticityResid_Def.hpp
    ${LCM_DIR}/evaluators/TLElasResid.hpp
    ${LCM_DIR}/evaluators/TLElasResid_Def.hpp
    ${LCM_DIR}/evaluators/EnergyPotential.hpp
    ${LCM_DIR}/evaluators/EnergyPotential_Def.hpp
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
    ${LCM_DIR}/evaluators/Localization.hpp
    ${LCM_DIR}/evaluators/Localization_Def.hpp
    ${LCM_DIR}/evaluators/SurfaceBasis.hpp
    ${LCM_DIR}/evaluators/SurfaceBasis_Def.hpp
    ${LCM_DIR}/evaluators/SurfaceScalarJump.hpp
    ${LCM_DIR}/evaluators/SurfaceScalarJump_Def.hpp
    ${LCM_DIR}/evaluators/SurfaceVectorJump.hpp
    ${LCM_DIR}/evaluators/SurfaceVectorJump_Def.hpp
    ${LCM_DIR}/evaluators/CurrentCoords.hpp
    ${LCM_DIR}/evaluators/CurrentCoords_Def.hpp
    ${LCM_DIR}/evaluators/DamageSource.hpp
    ${LCM_DIR}/evaluators/DamageSource_Def.hpp
    ${LCM_DIR}/evaluators/BulkModulus.hpp
    ${LCM_DIR}/evaluators/BulkModulus_Def.hpp
    ${LCM_DIR}/evaluators/ShearModulus.hpp
    ${LCM_DIR}/evaluators/ShearModulus_Def.hpp
    ${LCM_DIR}/evaluators/KfieldBC.hpp
    ${LCM_DIR}/evaluators/KfieldBC_Def.hpp
    ${LCM_DIR}/evaluators/TorsionBC.hpp
    ${LCM_DIR}/evaluators/TorsionBC_Def.hpp
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
    ${LCM_DIR}/evaluators/TimeDepBC.hpp
    ${LCM_DIR}/evaluators/TimeDepBC_Def.hpp
    ${LCM_DIR}/evaluators/DiffusionCoefficient.hpp
    ${LCM_DIR}/evaluators/DiffusionCoefficient_Def.hpp
    ${LCM_DIR}/evaluators/EquilibriumConstant.hpp
    ${LCM_DIR}/evaluators/EquilibriumConstant_Def.hpp
    ${LCM_DIR}/evaluators/TrappedSolvent.hpp
    ${LCM_DIR}/evaluators/TrappedSolvent_Def.hpp
    ${LCM_DIR}/evaluators/EffectiveDiffusivity.hpp
    ${LCM_DIR}/evaluators/EffectiveDiffusivity_Def.hpp
    ${LCM_DIR}/evaluators/TrappedConcentration.hpp
    ${LCM_DIR}/evaluators/TrappedConcentration_Def.hpp
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
    ${LCM_DIR}/evaluators/QptLocation.hpp
    ${LCM_DIR}/evaluators/QptLocation_Def.hpp
    ${LCM_DIR}/evaluators/MooneyRivlin.hpp
    ${LCM_DIR}/evaluators/MooneyRivlin_Def.hpp
    ${LCM_DIR}/evaluators/MooneyRivlinDamage.hpp
    ${LCM_DIR}/evaluators/MooneyRivlinDamage_Def.hpp
    ${LCM_DIR}/evaluators/MooneyRivlin_Incompressible.hpp
    ${LCM_DIR}/evaluators/MooneyRivlin_Incompressible_Def.hpp
    ${LCM_DIR}/evaluators/MooneyRivlin_Incompressible_Damage.hpp
    ${LCM_DIR}/evaluators/MooneyRivlin_Incompressible_Damage_Def.hpp
    ${LCM_DIR}/evaluators/RIHMR.hpp
    ${LCM_DIR}/evaluators/RIHMR_Def.hpp
    ${LCM_DIR}/evaluators/GursonHMR.hpp
    ${LCM_DIR}/evaluators/GursonHMR_Def.hpp
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
 )

  IF(ALBANY_MPI)
    SET(HEADERS ${HEADERS}
      ${LCM_DIR}/evaluators/MultiScaleStress.hpp
      ${LCM_DIR}/evaluators/MultiScaleStress_Def.hpp
    )
  ENDIF()

  #LCM utils
  SET(SOURCES ${SOURCES}
    ${LCM_DIR}/utils/Geometry.cc
    ${LCM_DIR}/utils/FloatingPoint.cc
    ${LCM_DIR}/utils/Partition.cc
    ${LCM_DIR}/utils/BarycentricSubdivision.cc
    ${LCM_DIR}/utils/Tensor.cc
    ${LCM_DIR}/utils/Topology.cc
    ${LCM_DIR}/utils/LocalNonlinearSolver.cc
    ${LCM_DIR}/utils/Projection.cpp
  )

  SET(HEADERS ${HEADERS}
    ${LCM_DIR}/utils/Geometry.h
    ${LCM_DIR}/utils/FloatingPoint.h
    ${LCM_DIR}/utils/Partition.h
    ${LCM_DIR}/utils/Tensor.h
    ${LCM_DIR}/utils/Topology.h
    ${LCM_DIR}/utils/LocalNonlinearSolver.h
    ${LCM_DIR}/utils/Projection.hpp
  )

IF(ALBANY_LAME OR ALBANY_LAMENT)
  SET(SOURCES ${SOURCES}
    ${LCM_DIR}/utils/LameUtils.cpp
    ${LCM_DIR}/problems/LameProblem.cpp
  )
  SET(HEADERS ${HEADERS}
    ${LCM_DIR}/utils/LameUtils.hpp
    ${LCM_DIR}/problems/LameProblem.hpp
  )
ENDIF()

IF(ALBANY_LAME)
  SET(SOURCES ${SOURCES}
    ${LCM_DIR}/evaluators/LameStress.cpp
  )
  SET(HEADERS ${HEADERS}
    ${LCM_DIR}/evaluators/LameStress.hpp
    ${LCM_DIR}/evaluators/LameStress_Def.hpp
  )
ENDIF()

IF(ALBANY_LAMENT)
  SET(SOURCES ${SOURCES}
    ${LCM_DIR}/evaluators/LamentStress.cpp
  )
  SET(HEADERS ${HEADERS}
    ${LCM_DIR}/evaluators/LamentStress.hpp
    ${LCM_DIR}/evaluators/LamentStress_Def.hpp
  )
ENDIF()

ENDIF()

# FELIX
IF(ALBANY_FELIX)
  SET(FELIX_DIR ${Albany_SOURCE_DIR}/src/FELIX)

  # FELIX evaluators
  SET(SOURCES ${SOURCES}
       ${FELIX_DIR}/evaluators/FELIX_StokesMomentumResid.cpp
       ${FELIX_DIR}/evaluators/FELIX_StokesFOResid.cpp
       ${FELIX_DIR}/evaluators/FELIX_Viscosity.cpp
       ${FELIX_DIR}/evaluators/FELIX_ViscosityFO.cpp
       ${FELIX_DIR}/evaluators/FELIX_StokesContinuityResid.cpp
       ${FELIX_DIR}/evaluators/FELIX_StokesTauM.cpp
       ${FELIX_DIR}/evaluators/FELIX_StokesContravarientMetricTensor.cpp
       ${FELIX_DIR}/evaluators/FELIX_StokesRm.cpp
       ${FELIX_DIR}/evaluators/FELIX_StokesBodyForce.cpp
       ${FELIX_DIR}/evaluators/FELIX_StokesFOBodyForce.cpp
   )
  SET(HEADERS ${HEADERS}
       ${FELIX_DIR}/evaluators/FELIX_StokesMomentumResid.hpp
       ${FELIX_DIR}/evaluators/FELIX_StokesFOResid_Def.hpp
       ${FELIX_DIR}/evaluators/FELIX_StokesFOResid.hpp
       ${FELIX_DIR}/evaluators/FELIX_StokesMomentumResid_Def.hpp
       ${FELIX_DIR}/evaluators/FELIX_ViscosityFO.hpp
       ${FELIX_DIR}/evaluators/FELIX_ViscosityFO_Def.hpp
       ${FELIX_DIR}/evaluators/FELIX_StokesContinuityResid.hpp
       ${FELIX_DIR}/evaluators/FELIX_StokesContinuityResid_Def.hpp
       ${FELIX_DIR}/evaluators/FELIX_StokesTauM.hpp
       ${FELIX_DIR}/evaluators/FELIX_StokesTauM_Def.hpp
       ${FELIX_DIR}/evaluators/FELIX_StokesContravarientMetricTensor.hpp
       ${FELIX_DIR}/evaluators/FELIX_StokesContravarientMetricTensor_Def.hpp
       ${FELIX_DIR}/evaluators/FELIX_StokesRm.hpp
       ${FELIX_DIR}/evaluators/FELIX_StokesRm_Def.hpp
       ${FELIX_DIR}/evaluators/FELIX_StokesBodyForce.hpp
       ${FELIX_DIR}/evaluators/FELIX_StokesBodyForce_Def.hpp
       ${FELIX_DIR}/evaluators/FELIX_StokesFOBodyForce.hpp
       ${FELIX_DIR}/evaluators/FELIX_StokesFOBodyForce_Def.hpp
  )
 # FELIX problems
  SET(SOURCES ${SOURCES}
       ${FELIX_DIR}/problems/FELIX_Stokes.cpp
       ${FELIX_DIR}/problems/FELIX_StokesFO.cpp
     )
  SET(HEADERS ${HEADERS}
       ${FELIX_DIR}/problems/FELIX_Stokes.hpp
       ${FELIX_DIR}/problems/FELIX_StokesFO.hpp
     )
ENDIF()

SET(SOURCES ${SOURCES}
       QCAD_Solver.cpp
       evaluators/QCAD_Permittivity.cpp
       evaluators/QCAD_PoissonResid.cpp
       evaluators/QCAD_PoissonSource.cpp
       evaluators/QCAD_PoissonDirichlet.cpp
       evaluators/QCAD_ResponseFieldIntegral.cpp
       evaluators/QCAD_ResponseFieldValue.cpp
       evaluators/QCAD_ResponseSaddleValue.cpp
       evaluators/QCAD_ResponseSaveField.cpp
       evaluators/QCAD_ResponseCenterOfMass.cpp
       evaluators/QCAD_ResponseRegionBoundary.cpp
       evaluators/QCAD_EvaluatorTools.cpp
       problems/QCAD_PoissonProblem.cpp
       problems/QCAD_MaterialDatabase.cpp
       responses/QCAD_SaddleValueResponseFunction.cpp
  )

SET(HEADERS ${HEADERS}
       QCAD_Solver.hpp
       evaluators/QCAD_Permittivity.hpp
       evaluators/QCAD_Permittivity_Def.hpp
       evaluators/QCAD_PoissonResid.hpp
       evaluators/QCAD_PoissonResid_Def.hpp
       evaluators/QCAD_PoissonSource.hpp
       evaluators/QCAD_PoissonSource_Def.hpp
       evaluators/QCAD_PoissonDirichlet.hpp
       evaluators/QCAD_PoissonDirichlet_Def.hpp
       evaluators/QCAD_ResponseFieldIntegral.hpp
       evaluators/QCAD_ResponseFieldIntegral_Def.hpp
       evaluators/QCAD_ResponseFieldValue.hpp
       evaluators/QCAD_ResponseFieldValue_Def.hpp
       evaluators/QCAD_ResponseSaddleValue.hpp
       evaluators/QCAD_ResponseSaddleValue_Def.hpp
       evaluators/QCAD_ResponseSaveField.hpp
       evaluators/QCAD_ResponseSaveField_Def.hpp
       evaluators/QCAD_ResponseCenterOfMass.hpp
       evaluators/QCAD_ResponseCenterOfMass_Def.hpp
       evaluators/QCAD_ResponseRegionBoundary.hpp
       evaluators/QCAD_ResponseRegionBoundary_Def.hpp
       evaluators/QCAD_EvaluatorTools.cpp
       evaluators/QCAD_EvaluatorTools.cpp
       evaluators/QCAD_EvaluatorTools.hpp
       evaluators/QCAD_EvaluatorTools_Def.hpp
       problems/QCAD_PoissonProblem.hpp
       responses/QCAD_SaddleValueResponseFunction.hpp
  )


#QCAD Schrodinger problem
SET(SOURCES ${SOURCES}
       evaluators/QCAD_SchrodingerPotential.cpp
       evaluators/QCAD_SchrodingerResid.cpp
       problems/QCAD_SchrodingerProblem.cpp
  )

SET(HEADERS ${HEADERS}
       evaluators/QCAD_SchrodingerPotential.hpp
       evaluators/QCAD_SchrodingerPotential_Def.hpp
       evaluators/QCAD_SchrodingerResid.hpp
       evaluators/QCAD_SchrodingerResid_Def.hpp
       problems/QCAD_SchrodingerProblem.hpp
  )


SET(MESHMOVER_DIR ${Albany_SOURCE_DIR}/src/MeshMover/src)
SET(CUTR_SRCS
  ${MESHMOVER_DIR}/CUTR_CubitMeshMover.cpp
  ${MESHMOVER_DIR}/MeshMorpher.cpp
  ${MESHMOVER_DIR}/MeshMorpher_XWR.cpp
  ${MESHMOVER_DIR}/MeshMorpher_BMS.cpp
  ${MESHMOVER_DIR}/MeshMorpher_smooth.cpp
  ${MESHMOVER_DIR}/MoverProblem.cpp
  ${MESHMOVER_DIR}/STKMeshData.cpp
  ${MESHMOVER_DIR}/ToyProblem.cpp
  ${MESHMOVER_DIR}/util/CubitBox.cpp
  ${MESHMOVER_DIR}/util/CubitMatrix.cpp
  ${MESHMOVER_DIR}/util/CubitPlane.cpp
  ${MESHMOVER_DIR}/util/CubitVector.cpp
  ${MESHMOVER_DIR}/util/GeoNode.cpp
  ${MESHMOVER_DIR}/util/GeoTet.cpp
  ${MESHMOVER_DIR}/util/MMGfxDebug.cpp
  ${MESHMOVER_DIR}/util/MemoryBlock.cpp
  ${MESHMOVER_DIR}/util/MemoryManager.cpp
  ${MESHMOVER_DIR}/util/KDDTree.cpp
  ${MESHMOVER_DIR}/util/KDDTreeNode.cpp
  ${MESHMOVER_DIR}/util/PriorityQueue.cpp
  ${MESHMOVER_DIR}/util/SimpleNode.cpp
  ${MESHMOVER_DIR}/util/TDDelaunay.cpp
  ${MESHMOVER_DIR}/util/TetFacetorTool.cpp
  ${MESHMOVER_DIR}/util/ToolData.cpp
  ${MESHMOVER_DIR}/util/ToolDataUser.cpp
  )

SET(CUTR_HDRS
  ${MESHMOVER_DIR}/CUTR_CubitMeshMover.hpp
  ${MESHMOVER_DIR}/MeshMorpher.hpp
  ${MESHMOVER_DIR}/MeshMorpher_XWR.hpp
  ${MESHMOVER_DIR}/MeshMorpher_BMS.hpp
  ${MESHMOVER_DIR}/MeshMorpher_smooth.hpp
  ${MESHMOVER_DIR}/MoverProblem.hpp
  ${MESHMOVER_DIR}/STKMeshData.hpp
  ${MESHMOVER_DIR}/ToyProblem.hpp
  ${MESHMOVER_DIR}/util/CubitBox.cpp
  ${MESHMOVER_DIR}/util/CubitMatrix.cpp
  ${MESHMOVER_DIR}/util/CubitPlane.cpp
  ${MESHMOVER_DIR}/util/CubitVector.cpp
  ${MESHMOVER_DIR}/util/GeoNode.cpp
  ${MESHMOVER_DIR}/util/GeoTet.cpp
  ${MESHMOVER_DIR}/util/MMGfxDebug.cpp
  ${MESHMOVER_DIR}/util/MemoryBlock.cpp
  ${MESHMOVER_DIR}/util/MemoryManager.cpp
  ${MESHMOVER_DIR}/util/KDDTree.cpp
  ${MESHMOVER_DIR}/util/KDDTreeNode.cpp
  ${MESHMOVER_DIR}/util/PriorityQueue.cpp
  ${MESHMOVER_DIR}/util/SimpleNode.cpp
  ${MESHMOVER_DIR}/util/TDDelaunay.cpp
  ${MESHMOVER_DIR}/util/TetFacetorTool.cpp
  ${MESHMOVER_DIR}/util/ToolData.cpp
  ${MESHMOVER_DIR}/util/ToolDataUser.cpp
  )

set(ALB_TRILINOS_LIBS ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})

include_directories (${Trilinos_INCLUDE_DIRS}
                     ./ ./evaluators ./problems ./responses ./stk
                     ./LCM/ ./LCM/evaluators ./LCM/utils ./FELIX/problems
                     ./FELIX/evaluators
                     ${Trilinos_TPL_INCLUDE_DIRS})

link_directories (${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})

# LAME
IF(ALBANY_LAME)
  include_directories (${LAME_INCLUDE_DIR})
  link_directories (${LAME_LIBRARY_DIR})
  set(LAME_LIB lame)
ELSE()
  set(LAME_LIB )
ENDIF()

# LAMENT
IF(ALBANY_LAMENT)
  include_directories (${LAMENT_INCLUDE_DIR})
  link_directories (${LAMENT_LIBRARY_DIR})
  set(LAMENT_LIB lament)
ELSE()
  set(LAMENT_LIB )
ENDIF()

# Albany CI (Configuration Interaction)
IF(ALBANY_CI)
  include_directories (${ALBANY_CI_INCLUDE_DIR})
  link_directories (${ALBANY_CI_LIBRARY_DIR})
  set(ALBANY_CI_LIB albanyCILib)
ELSE()
  set(ALBANY_CI_LIB )
ENDIF()

if (ALBANY_ENABLE_CUBIT)
  INCLUDE(${MESHMOVER_DIR}/UseUtil.cmake)
  set (SOURCES ${SOURCES} ${CUTR_SRCS})
  set (HEADERS ${HEADERS} ${CUTR_HDRS})
  include_directories (${MESHMOVER_DIR} ${MESHMOVER_DIR}/../cubit)
  link_directories (${MESHMOVER_DIR}/../cubit/linux64)
  set(ALB_CUBIT_LIBS  cubiti19)
  ADD_DEFINITIONS(-DALBANY_CUTR)
  MESSAGE("-- CUBIT is Enabled, compiling with -DALBANY_CUTR")
endif (ALBANY_ENABLE_CUBIT)

add_library(albanyLib ${SOURCES} ${HEADERS})
# Now build executables from this library, tailored main()s, and Trilinos

add_executable(Albany Main_Solve.cpp)
add_executable(AlbanySG Main_SGSolve.cpp)
if (ALBANY_ASCR)
  add_executable(AlbanyAdjoint Main_Adjoint.cpp)
  add_executable(AlbanySGAdjoint Main_SGAdjoint.cpp)
  add_executable(AlbanyCoupled Main_Coupled.cpp)
  add_executable(AlbanySGCoupled Main_SGCoupled.cpp)
endif()
IF (ALBANY_DAKOTA)
  add_executable(AlbanyDakota Main_Dakota.cpp Albany_Dakota.cpp Albany_Dakota.hpp)
ENDIF ()

add_executable(AlbanyAnalysis Main_Analysis.cpp)

# Tests
IF (ALBANY_LCM)
  add_executable(PartitionTest LCM/test/utils/PartitionTest.cc)
  add_executable(NodeUpdate LCM/test/utils/NodeUpdate.cc)
  add_executable(Subdivision LCM/test/utils/Subdivision.cc)
  add_executable(MeshComponents LCM/test/utils/MeshComponents.cc)
  add_executable(
    utLocalNonlinearSolver
    LCM/test/unit_tests/StandardUnitTestMain.cpp
    LCM/test/unit_tests/utLocalNonlinearSolver.cpp
  )
  add_executable(
    utLocalization
    LCM/test/unit_tests/StandardUnitTestMain.cpp
    LCM/test/unit_tests/utLocalization.cpp
  )
  add_executable(
    utTensorUtils
    LCM/test/unit_tests/StandardUnitTestMain.cpp
    LCM/test/unit_tests/utTensorUtils.cpp
  )
  add_executable(
    utProjection
    LCM/test/unit_tests/StandardUnitTestMain.cpp
    LCM/test/unit_tests/utProjection.cpp
  )
ENDIF ()
IF (ALBANY_LAME OR ALBANY_LAMENT)
  add_executable(utLameStress_elastic
    PHAL_Dimension.cpp
    PHAL_AlbanyTraits.cpp
    Albany_Utils.cpp
    Albany_StateManager.cpp
    stk/Albany_GenericSTKMeshStruct.cpp
    stk/Albany_TmplSTKMeshStruct.cpp
    stk/Albany_STKDiscretization.cpp
    problems/QCAD_MaterialDatabase.cpp
    ${LCM_DIR}/utils/Tensor.cc
    ${LCM_DIR}/utils/LameUtils.cpp
    ${LCM_DIR}/evaluators/LameStress.cpp
    ${LCM_DIR}/evaluators/SetField.cpp
    ${LCM_DIR}/test/unit_tests/StandardUnitTestMain.cpp
    ${LCM_DIR}/test/unit_tests/utLameStress_elastic.cpp
  )
  target_link_libraries(utLameStress_elastic ${LAME_LIB} ${LAMENT_LIB} ${Trilinos_LIBRARIES})
ENDIF ()

set(ALL_LIBRARIES ${ALBANY_CI_LIB} ${ALB_TRILINOS_LIBS} ${ALB_CUBIT_LIBS} ${LAME_LIB} ${LAMENT_LIB} ${Trilinos_EXTRA_LD_FLAGS} ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES} )

target_link_libraries(Albany          albanyLib ${ALL_LIBRARIES})
target_link_libraries(AlbanySG        albanyLib ${ALL_LIBRARIES})
if (ALBANY_ASCR)
  target_link_libraries(AlbanyAdjoint   albanyLib ${ALL_LIBRARIES})
  target_link_libraries(AlbanySGAdjoint   albanyLib ${ALL_LIBRARIES})
  target_link_libraries(AlbanyCoupled   albanyLib ${ALL_LIBRARIES})
  target_link_libraries(AlbanySGCoupled   albanyLib ${ALL_LIBRARIES})
endif()
IF (ALBANY_DAKOTA)
  target_link_libraries(AlbanyDakota    albanyLib ${ALL_LIBRARIES})
ENDIF ()

target_link_libraries(AlbanyAnalysis  albanyLib ${ALL_LIBRARIES})

#Linking tests
IF (ALBANY_LCM)
  target_link_libraries(PartitionTest albanyLib ${ALL_LIBRARIES})
  target_link_libraries(NodeUpdate albanyLib ${ALL_LIBRARIES})
  target_link_libraries(Subdivision albanyLib ${ALL_LIBRARIES})
  target_link_libraries(MeshComponents albanyLib ${ALL_LIBRARIES})
  target_link_libraries(utLocalNonlinearSolver albanyLib ${ALL_LIBRARIES})
  target_link_libraries(utLocalization albanyLib ${ALL_LIBRARIES})
  target_link_libraries(utTensorUtils albanyLib ${ALL_LIBRARIES})
  target_link_libraries(utProjection albanyLib ${ALL_LIBRARIES})
ENDIF ()
