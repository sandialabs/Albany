##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//


INCLUDE(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${Trilinos_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LIBRARIES ${Trilinos_LIBRARY_DIRS})
set( CMAKE_REQUIRED_FLAGS ${Trilinos_CXX_FLAGS} )


SET(SOURCES
  Albany_SolverFactory.cpp
  Albany_Utils.cpp
  PHAL_AlbanyTraits.cpp
  PHAL_Dimension.cpp
  Albany_Application.cpp
  Albany_Memory.cpp
  Albany_ModelFactory.cpp
  Albany_ModelEvaluatorT.cpp
  Albany_NullSpaceUtils.cpp
  Albany_ObserverImpl.cpp
  Albany_PiroObserverT.cpp
  Albany_StatelessObserverImpl.cpp
  Albany_StateManager.cpp
  PHAL_Utilities.cpp
  )

#IKT, FIXME: remove OR ALBANY_ATO from following if when 
#ATO is free of Epetra 
IF (ALBANY_EPETRA OR ALBANY_ATO)
  SET(SOURCES ${SOURCES} Albany_ObserverFactory.cpp)
  IF(ALBANY_64BIT_INT)
    SET(SOURCES ${SOURCES} Petra_Converters_64.cpp)
  ELSE()
    SET(SOURCES ${SOURCES} Petra_Converters.cpp)
  ENDIF()
ENDIF()


IF(ALBANY_EPETRA)
  SET(SOURCES ${SOURCES}
    Albany_ModelEvaluator.cpp
    Albany_NOXObserver.cpp
    Albany_NOXStatelessObserver.cpp
    Albany_PiroObserver.cpp
    Albany_RythmosObserver.cpp
    Albany_SaveEigenData.cpp
  )
ENDIF()

SET(HEADERS
  Albany_Application.hpp
  Albany_DataTypes.hpp
  Albany_DistributedParameterLibrary.hpp
  Albany_DistributedParameterDerivativeOpT.hpp
  Albany_DistributedParameterLibrary_Tpetra.hpp
  Albany_DummyParameterAccessor.hpp
  Albany_EigendataInfoStructT.hpp
  Albany_Memory.hpp
  Albany_ModelFactory.hpp
  Albany_ModelEvaluatorT.hpp
  Albany_NullSpaceUtils.hpp
  Albany_ObserverImpl.hpp
  Albany_PiroObserverT.hpp
  Albany_SolverFactory.hpp
  Albany_StateManager.hpp
  Albany_StateInfoStruct.hpp
  Albany_StatelessObserverImpl.hpp
  Albany_Utils.hpp
  PHAL_AlbanyTraits.hpp
  PHAL_Dimension.hpp
  PHAL_FactoryTraits.hpp
  PHAL_TypeKeyMap.hpp
  PHAL_Utilities.hpp
  PHAL_Utilities_Def.hpp
  PHAL_Workset.hpp
  )

IF(ALBANY_EPETRA)
  SET(HEADERS ${HEADERS}
    Albany_EigendataInfoStruct.hpp
    Albany_SaveEigenData.hpp
    Albany_ModelEvaluator.hpp
    Albany_NOXObserver.hpp
    Albany_NOXStatelessObserver.hpp
    Albany_ObserverFactory.hpp
    Albany_PiroObserver.hpp
    Albany_RythmosObserver.hpp
    Petra_Converters.hpp
  )
ENDIF()

#utility
SET(SOURCES ${SOURCES}
  utility/Counter.cpp
  utility/CounterMonitor.cpp
  utility/DisplayTable.cpp
  utility/PerformanceContext.cpp
  utility/TimeMonitor.cpp
  utility/VariableMonitor.cpp
  utility/StaticAllocator.cpp
  )
SET(HEADERS ${HEADERS}
  utility/Counter.hpp
  utility/CounterMonitor.hpp
  utility/DisplayTable.hpp
  utility/MonitorBase.hpp
  utility/PerformanceContext.hpp
  utility/string.hpp
  utility/TimeGuard.hpp
  utility/TimeMonitor.hpp
  utility/VariableMonitor.hpp
  utility/StaticAllocator.hpp
  utility/math/Tensor.hpp
  utility/math/TensorCommon.hpp
  utility/math/TensorDetail.hpp
  utility/math/TensorImpl.hpp
  )

#problems
SET(SOURCES ${SOURCES}
  problems/Albany_AbstractProblem.cpp
  problems/Albany_BCUtils.cpp
  problems/Albany_EvaluatorUtils.cpp
  problems/Albany_HeatProblem.cpp
  problems/Albany_Layouts.cpp
  problems/Albany_ProblemFactory.cpp
  problems/Albany_ProblemUtils.cpp
  problems/Albany_ResponseUtilities.cpp
  problems/QCAD_MaterialDatabase.cpp
  )
SET(PDE_SRCS
  problems/Albany_AdvDiffProblem.cpp
  problems/Albany_CahnHillProblem.cpp
  problems/Albany_ComprNSProblem.cpp
  problems/Albany_GPAMProblem.cpp
  problems/Albany_Helmholtz2DProblem.cpp
  problems/Albany_LinComprNSProblem.cpp
  problems/Albany_ODEProblem.cpp
  problems/Albany_NavierStokes.cpp
  problems/Albany_PNPProblem.cpp
  )
SET(HEADERS ${HEADERS}
  problems/Albany_AbstractProblem.hpp
  problems/Albany_BCUtils.hpp
  problems/Albany_BCUtils_Def.hpp
  problems/Albany_EvaluatorUtils.hpp
  problems/Albany_EvaluatorUtils_Def.hpp
  problems/Albany_HeatProblem.hpp
  problems/Albany_Layouts.hpp
  problems/Albany_ProblemFactory.hpp
  problems/Albany_ProblemUtils.hpp
  problems/Albany_ResponseUtilities.hpp
  problems/Albany_ResponseUtilities_Def.hpp
  problems/QCAD_MaterialDatabase.hpp
  )
SET(PDE_HDRS
  problems/Albany_AdvDiffProblem.hpp
  problems/Albany_CahnHillProblem.hpp
  problems/Albany_ComprNSProblem.hpp
  problems/Albany_GPAMProblem.hpp
  problems/Albany_Helmholtz2DProblem.hpp
  problems/Albany_LinComprNSProblem.hpp
  problems/Albany_NavierStokes.hpp
  problems/Albany_ODEProblem.hpp
  problems/Albany_PNPProblem.hpp
  )

#responses
IF (ALBANY_EPETRA)
SET(SOURCES ${SOURCES}
  responses/Epetra_GatherAllV.cpp
)
SET(HEADERS ${HEADERS}
  responses/Epetra_GatherAllV.hpp
)
ENDIF()
SET(SOURCES ${SOURCES}
  responses/Albany_AggregateScalarResponseFunction.cpp
  responses/Albany_CumulativeScalarResponseFunction.cpp
  responses/Albany_DistributedResponseFunction.cpp
  responses/Albany_FieldManagerScalarResponseFunction.cpp
  responses/Albany_FieldManagerResidualOnlyResponseFunction.cpp
  responses/Albany_SamplingBasedScalarResponseFunction.cpp
  responses/Albany_KLResponseFunction.cpp
  responses/Albany_ResponseFactory.cpp
  responses/Albany_ScalarResponseFunction.cpp
  responses/Albany_SolutionCullingStrategy.cpp
  responses/Albany_SolutionAverageResponseFunction.cpp
  responses/Albany_SolutionFileResponseFunction.cpp
  responses/Albany_SolutionMaxValueResponseFunction.cpp
  responses/Albany_SolutionMinValueResponseFunction.cpp
  responses/Albany_SolutionTwoNormResponseFunction.cpp
  responses/Albany_SolutionValuesResponseFunction.cpp
  )
SET(HEADERS ${HEADERS}
  responses/Albany_AbstractResponseFunction.hpp
  responses/Albany_AggregateScalarResponseFunction.hpp
  responses/Albany_CumulativeScalarResponseFunction.hpp
  responses/Albany_DistributedResponseFunction.hpp
  responses/Albany_FieldManagerScalarResponseFunction.hpp
  responses/Albany_KLResponseFunction.hpp
  responses/Albany_SamplingBasedScalarResponseFunction.hpp
  responses/Albany_ResponseFactory.hpp
  responses/Albany_ScalarResponseFunction.hpp
  responses/Albany_SolutionAverageResponseFunction.hpp
  responses/Albany_SolutionCullingStrategy.hpp
  responses/Albany_SolutionFileResponseFunction.hpp
  responses/Albany_SolutionMaxValueResponseFunction.hpp
  responses/Albany_SolutionTwoNormResponseFunction.hpp
  responses/Albany_SolutionValuesResponseFunction.hpp
  )

IF(ALBANY_EPETRA)

SET(SOURCES ${SOURCES}
  responses/Albany_SolutionResponseFunction.cpp
)

SET(HEADERS ${HEADERS}
  responses/Albany_SolutionResponseFunction.hpp
)

ENDIF()

IF(ALBANY_EPETRA)
SET(SOURCES ${SOURCES}
  responses/Epetra_ImportWithAlternateMap.cpp
)
SET(HEADERS ${HEADERS}
  responses/Epetra_ImportWithAlternateMap.hpp
)
ENDIF()

#discretization
SET(SOURCES ${SOURCES}
  disc/Adapt_NodalDataBase.cpp
  disc/Adapt_NodalDataVector.cpp
  disc/Albany_AbstractMeshStruct.cpp
  disc/Albany_DiscretizationFactory.cpp
  )
SET(HEADERS ${HEADERS}
  disc/Adapt_NodalDataBase.hpp
  disc/Adapt_NodalDataVector.hpp
  disc/Adapt_NodalFieldUtils.hpp
  disc/Albany_AbstractDiscretization.hpp
  disc/Albany_AbstractFieldContainer.hpp
  disc/Albany_AbstractMeshStruct.hpp
  disc/Albany_AbstractNodeFieldContainer.hpp
  disc/Albany_DiscretizationFactory.hpp
  disc/Albany_NodalDOFManager.hpp
  )

#evaluators
SET(SOURCES ${SOURCES}
  evaluators/Adapt_ElementSizeField.cpp
  evaluators/PHAL_Absorption.cpp
  evaluators/PHAL_AddNoise.cpp
  evaluators/PHAL_ComputeBasisFunctions.cpp
  evaluators/PHAL_ComputeBasisFunctionsSide.cpp
  evaluators/PHAL_Dirichlet.cpp
  evaluators/PHAL_DirichletCoordinateFunction.cpp
  evaluators/PHAL_DirichletOffNodeSet.cpp
  evaluators/PHAL_DirichletField.cpp
  evaluators/PHAL_DOFCellToSide.cpp
  evaluators/PHAL_DOFCellToSideQP.cpp
  evaluators/PHAL_DOFInterpolation.cpp
  evaluators/PHAL_DOFInterpolationSide.cpp
  evaluators/PHAL_DOFGradInterpolation.cpp
  evaluators/PHAL_DOFGradInterpolationSide.cpp
  evaluators/PHAL_DOFDivInterpolationSide.cpp
  evaluators/PHAL_DOFSideToCell.cpp
  evaluators/PHAL_DOFVecGradInterpolation.cpp
  evaluators/PHAL_DOFVecGradInterpolationSide.cpp
  evaluators/PHAL_DOFTensorGradInterpolation.cpp
  evaluators/PHAL_DOFTensorInterpolation.cpp
  evaluators/PHAL_DOFVecInterpolation.cpp
  evaluators/PHAL_DOFVecInterpolationSide.cpp
  evaluators/PHAL_GatherAuxData.cpp
  evaluators/PHAL_GatherCoordinateVector.cpp
  evaluators/PHAL_GatherScalarNodalParameter.cpp
  evaluators/PHAL_ScatterScalarNodalParameter.cpp
  evaluators/PHAL_GatherSolution.cpp
  evaluators/PHAL_HeatEqResid.cpp
  evaluators/PHAL_IdentityCoordinateFunctionTraits.cpp
  evaluators/PHAL_LoadSideSetStateField.cpp
  evaluators/PHAL_LoadStateField.cpp
  evaluators/PHAL_ConvertFieldType.cpp
  evaluators/PHAL_MapToPhysicalFrame.cpp
  evaluators/PHAL_MapToPhysicalFrameSide.cpp
  evaluators/PHAL_Neumann.cpp
  evaluators/PHAL_NodesToCellInterpolation.cpp
  evaluators/PHAL_NSMaterialProperty.cpp
  evaluators/PHAL_QuadPointsToCellInterpolation.cpp
  evaluators/PHAL_ResponseFieldIntegralT.cpp
  evaluators/PHAL_ResponseThermalEnergyT.cpp
  evaluators/PHAL_SaveCellStateField.cpp
  evaluators/PHAL_SaveNodalField.cpp
  evaluators/PHAL_SaveSideSetStateField.cpp
  evaluators/PHAL_SaveStateField.cpp
  evaluators/PHAL_ScatterResidual.cpp
  evaluators/PHAL_ScatterScalarResponse.cpp
  evaluators/PHAL_SeparableScatterScalarResponseT.cpp
  evaluators/PHAL_SharedParameter.cpp
  evaluators/PHAL_SideQuadPointsToSideInterpolation.cpp
  evaluators/PHAL_Source.cpp
  evaluators/PHAL_ResponseSquaredL2Error.cpp
  evaluators/PHAL_ResponseSquaredL2ErrorSide.cpp
  evaluators/PHAL_ThermalConductivity.cpp
  evaluators/QCAD_EvaluatorTools.cpp
  evaluators/QCAD_MathVector.cpp
  evaluators/QCAD_MeshRegion.cpp
  evaluators/QCAD_ResponseCenterOfMass.cpp
  evaluators/QCAD_ResponseFieldAverage.cpp
  evaluators/QCAD_ResponseFieldIntegral.cpp
  evaluators/QCAD_ResponseFieldValue.cpp
  evaluators/QCAD_ResponseSaveField.cpp
  )

SET(SOURCES ${SOURCES}
 evaluators/PHAL_ResponseFieldIntegral.cpp
 evaluators/PHAL_SeparableScatterScalarResponse.cpp
 )

IF (ALBANY_EPETRA)
  SET(SOURCES ${SOURCES}
    evaluators/PHAL_GatherEigenData.cpp
    evaluators/PHAL_GatherEigenvectors.cpp
  )
ENDIF()

SET(PDE_SRCS ${PDE_SRCS}
  evaluators/PHAL_AdvDiffResid.cpp
  evaluators/PHAL_CahnHillChemTerm.cpp
  evaluators/PHAL_CahnHillRhoResid.cpp
  evaluators/PHAL_CahnHillWResid.cpp
  evaluators/PHAL_ComprNSBodyForce.cpp
  evaluators/PHAL_ComprNSResid.cpp
  evaluators/PHAL_ComprNSViscosity.cpp
  evaluators/PHAL_Constant.cpp
  evaluators/PHAL_GPAMResid.cpp
  evaluators/PHAL_HelmholtzResid.cpp
  evaluators/PHAL_LangevinNoiseTerm.cpp
  evaluators/PHAL_LinComprNSBodyForce.cpp
  evaluators/PHAL_LinComprNSResid.cpp
  evaluators/PHAL_NSBodyForce.cpp
  evaluators/PHAL_NSContinuityResid.cpp
  evaluators/PHAL_NSContravarientMetricTensor.cpp
  evaluators/PHAL_NSForchheimerTerm.cpp
  evaluators/PHAL_NSMomentumResid.cpp
  evaluators/PHAL_NSNeutronEqResid.cpp
  evaluators/PHAL_NSPermeabilityTerm.cpp
  evaluators/PHAL_NSRm.cpp
  evaluators/PHAL_NSTauM.cpp
  evaluators/PHAL_NSTauT.cpp
  evaluators/PHAL_NSThermalEqResid.cpp
  evaluators/PHAL_ODEResid.cpp
  evaluators/PHAL_Permittivity.cpp
  evaluators/PNP_ConcentrationResid.cpp
  evaluators/PNP_PotentialResid.cpp
  )

SET(HEADERS ${HEADERS}
  evaluators/Adapt_ElementSizeField.hpp
  evaluators/Adapt_ElementSizeField_Def.hpp
  evaluators/PHAL_Absorption.hpp
  evaluators/PHAL_Absorption_Def.hpp
  evaluators/PHAL_AddNoise.hpp
  evaluators/PHAL_AddNoise_Def.hpp
  evaluators/PHAL_Constant.hpp
  evaluators/PHAL_Constant_Def.hpp
  evaluators/PHAL_Dirichlet.hpp
  evaluators/PHAL_Dirichlet_Def.hpp
  evaluators/PHAL_DirichletCoordinateFunction.hpp
  evaluators/PHAL_DirichletCoordinateFunction_Def.hpp
  evaluators/PHAL_DirichletField.hpp
  evaluators/PHAL_DirichletField_Def.hpp
  evaluators/PHAL_DirichletOffNodeSet.hpp
  evaluators/PHAL_DirichletOffNodeSet_Def.hpp
  evaluators/PHAL_DOFCellToSide.hpp
  evaluators/PHAL_DOFCellToSide_Def.hpp
  evaluators/PHAL_DOFCellToSideQP.hpp
  evaluators/PHAL_DOFCellToSideQP_Def.hpp
  evaluators/PHAL_DOFGradInterpolation.hpp
  evaluators/PHAL_DOFGradInterpolation_Def.hpp
  evaluators/PHAL_DOFGradInterpolationSide.hpp
  evaluators/PHAL_DOFGradInterpolationSide_Def.hpp
  evaluators/PHAL_DOFDivInterpolationSide.hpp
  evaluators/PHAL_DOFDivInterpolationSide_Def.hpp
  evaluators/PHAL_DOFInterpolation.hpp
  evaluators/PHAL_DOFInterpolation_Def.hpp
  evaluators/PHAL_DOFInterpolationSide.hpp
  evaluators/PHAL_DOFInterpolationSide_Def.hpp
  evaluators/PHAL_DOFTensorGradInterpolation.hpp
  evaluators/PHAL_DOFTensorGradInterpolation_Def.hpp
  evaluators/PHAL_DOFTensorInterpolation.hpp
  evaluators/PHAL_DOFTensorInterpolation_Def.hpp
  evaluators/PHAL_DOFSideToCell.hpp
  evaluators/PHAL_DOFSideToCell_Def.hpp
  evaluators/PHAL_DOFVecGradInterpolation.hpp
  evaluators/PHAL_DOFVecGradInterpolation_Def.hpp
  evaluators/PHAL_DOFVecGradInterpolationSide.hpp
  evaluators/PHAL_DOFVecGradInterpolationSide_Def.hpp
  evaluators/PHAL_DOFVecInterpolation.hpp
  evaluators/PHAL_DOFVecInterpolation_Def.hpp
  evaluators/PHAL_DOFVecInterpolationSide.hpp
  evaluators/PHAL_DOFVecInterpolationSide_Def.hpp
  evaluators/PHAL_GatherAuxData.hpp
  evaluators/PHAL_GatherAuxData_Def.hpp
  evaluators/PHAL_GatherCoordinateVector.hpp
  evaluators/PHAL_GatherCoordinateVector_Def.hpp
  evaluators/PHAL_GatherScalarNodalParameter.hpp
  evaluators/PHAL_GatherScalarNodalParameter_Def.hpp
  evaluators/PHAL_ScatterScalarNodalParameter.hpp
  evaluators/PHAL_ScatterScalarNodalParameter_Def.hpp
  evaluators/PHAL_GatherSolution.hpp
  evaluators/PHAL_GatherSolution_Def.hpp
  evaluators/PHAL_HeatEqResid.hpp
  evaluators/PHAL_HeatEqResid_Def.hpp
  evaluators/PHAL_IdentityCoordinateFunctionTraits.hpp
  evaluators/PHAL_IdentityCoordinateFunctionTraits_Def.hpp
  evaluators/PHAL_LoadSideSetStateField.hpp
  evaluators/PHAL_LoadSideSetStateField_Def.hpp
  evaluators/PHAL_LoadStateField.hpp
  evaluators/PHAL_LoadStateField_Def.hpp
  evaluators/PHAL_ConvertFieldType.hpp
  evaluators/PHAL_ConvertFieldType_Def.hpp
  evaluators/PHAL_MapToPhysicalFrame.hpp
  evaluators/PHAL_MapToPhysicalFrame_Def.hpp
  evaluators/PHAL_MapToPhysicalFrameSide.hpp
  evaluators/PHAL_MapToPhysicalFrameSide_Def.hpp
  evaluators/PHAL_Neumann.hpp
  evaluators/PHAL_Neumann_Def.hpp
  evaluators/PHAL_NodesToCellInterpolation.hpp
  evaluators/PHAL_NodesToCellInterpolation_Def.hpp
  evaluators/PHAL_NSMaterialProperty.hpp
  evaluators/PHAL_NSMaterialProperty_Def.hpp
  evaluators/PHAL_QuadPointsToCellInterpolation.hpp
  evaluators/PHAL_QuadPointsToCellInterpolation_Def.hpp
  evaluators/PHAL_ResponseFieldIntegralT.hpp
  evaluators/PHAL_ResponseFieldIntegralT_Def.hpp
  evaluators/PHAL_ResponseSquaredL2Error.hpp
  evaluators/PHAL_ResponseSquaredL2Error_Def.hpp
  evaluators/PHAL_ResponseSquaredL2ErrorSide.hpp
  evaluators/PHAL_ResponseSquaredL2ErrorSide_Def.hpp
  evaluators/PHAL_ResponseThermalEnergyT.hpp
  evaluators/PHAL_ResponseThermalEnergyT_Def.hpp
  evaluators/PHAL_SaveCellStateField.hpp
  evaluators/PHAL_SaveCellStateField_Def.hpp
  evaluators/PHAL_SaveNodalField.hpp
  evaluators/PHAL_SaveNodalField_Def.hpp
  evaluators/PHAL_SaveSideSetStateField.hpp
  evaluators/PHAL_SaveSideSetStateField_Def.hpp
  evaluators/PHAL_SaveStateField.hpp
  evaluators/PHAL_SaveStateField_Def.hpp
  evaluators/PHAL_ScatterResidual.hpp
  evaluators/PHAL_ScatterResidual_Def.hpp
  evaluators/PHAL_ScatterScalarResponse.hpp
  evaluators/PHAL_ScatterScalarResponse_Def.hpp
  evaluators/PHAL_SeparableScatterScalarResponseT.hpp
  evaluators/PHAL_SeparableScatterScalarResponseT_Def.hpp
  evaluators/PHAL_SharedParameter.hpp
  evaluators/PHAL_SharedParameter_Def.hpp
  evaluators/PHAL_SideQuadPointsToSideInterpolation.hpp
  evaluators/PHAL_SideQuadPointsToSideInterpolation_Def.hpp
  evaluators/PHAL_Source.hpp
  evaluators/PHAL_Source_Def.hpp
  evaluators/PHAL_ThermalConductivity.hpp
  evaluators/PHAL_ThermalConductivity_Def.hpp
  evaluators/QCAD_EvaluatorTools.hpp
  evaluators/QCAD_EvaluatorTools_Def.hpp
  evaluators/QCAD_MathVector.hpp
  evaluators/QCAD_MeshRegion.hpp
  evaluators/QCAD_MeshRegion_Def.hpp
  evaluators/QCAD_ResponseCenterOfMass.hpp
  evaluators/QCAD_ResponseCenterOfMass_Def.hpp
  evaluators/QCAD_ResponseFieldAverage.hpp
  evaluators/QCAD_ResponseFieldAverage_Def.hpp
  evaluators/QCAD_ResponseFieldIntegral.hpp
  evaluators/QCAD_ResponseFieldIntegral_Def.hpp
  evaluators/QCAD_ResponseFieldValue.hpp
  evaluators/QCAD_ResponseFieldValue_Def.hpp
  evaluators/QCAD_ResponseSaveField.hpp
  evaluators/QCAD_ResponseSaveField_Def.hpp
  )

IF (ALBANY_EPETRA)
  SET(HEADERS ${HEADERS}
    evaluators/PHAL_GatherEigenvectors.hpp
    evaluators/PHAL_GatherEigenvectors_Def.hpp
    evaluators/PHAL_GatherEigenData.hpp
    evaluators/PHAL_GatherEigenData_Def.hpp
    evaluators/PHAL_ResponseFieldIntegral.hpp
    evaluators/PHAL_ResponseFieldIntegral_Def.hpp
    evaluators/PHAL_SeparableScatterScalarResponse.hpp
    evaluators/PHAL_SeparableScatterScalarResponse_Def.hpp
  )
ENDIF()

SET(PDE_HDRS ${PDE_HDRS}
  evaluators/PHAL_AdvDiffResid.hpp
  evaluators/PHAL_AdvDiffResid_Def.hpp
  evaluators/PHAL_CahnHillRhoResid.hpp
  evaluators/PHAL_CahnHillRhoResid_Def.hpp
  evaluators/PHAL_CahnHillWResid.hpp
  evaluators/PHAL_CahnHillWResid_Def.hpp
  evaluators/PHAL_CahnHillChemTerm.hpp
  evaluators/PHAL_CahnHillChemTerm_Def.hpp
  evaluators/PHAL_ComprNSBodyForce.hpp
  evaluators/PHAL_ComprNSBodyForce_Def.hpp
  evaluators/PHAL_ComprNSResid.hpp
  evaluators/PHAL_ComprNSResid_Def.hpp
  evaluators/PHAL_ComprNSViscosity.hpp
  evaluators/PHAL_ComprNSViscosity_Def.hpp
  evaluators/PHAL_ComputeBasisFunctions.hpp
  evaluators/PHAL_ComputeBasisFunctions_Def.hpp
  evaluators/PHAL_ComputeBasisFunctionsSide.hpp
  evaluators/PHAL_ComputeBasisFunctionsSide_Def.hpp
  evaluators/PHAL_GPAMResid.hpp
  evaluators/PHAL_GPAMResid_Def.hpp
  evaluators/PHAL_HelmholtzResid.hpp
  evaluators/PHAL_HelmholtzResid_Def.hpp
  evaluators/PHAL_LangevinNoiseTerm.hpp
  evaluators/PHAL_LangevinNoiseTerm_Def.hpp
  evaluators/PHAL_LinComprNSBodyForce.hpp
  evaluators/PHAL_LinComprNSBodyForce_Def.hpp
  evaluators/PHAL_LinComprNSResid.hpp
  evaluators/PHAL_LinComprNSResid_Def.hpp
  evaluators/PHAL_NSContinuityResid.hpp
  evaluators/PHAL_NSContinuityResid_Def.hpp
  evaluators/PHAL_NSBodyForce.hpp
  evaluators/PHAL_NSBodyForce_Def.hpp
  evaluators/PHAL_NSContravarientMetricTensor.hpp
  evaluators/PHAL_NSContravarientMetricTensor_Def.hpp
  evaluators/PHAL_NSForchheimerTerm.hpp
  evaluators/PHAL_NSForchheimerTerm_Def.hpp
  evaluators/PHAL_NSMomentumResid.hpp
  evaluators/PHAL_NSMomentumResid_Def.hpp
  evaluators/PHAL_NSNeutronEqResid.hpp
  evaluators/PHAL_NSNeutronEqResid_Def.hpp
  evaluators/PHAL_NSPermeabilityTerm.hpp
  evaluators/PHAL_NSPermeabilityTerm_Def.hpp
  evaluators/PHAL_NSRm.hpp
  evaluators/PHAL_NSRm_Def.hpp
  evaluators/PHAL_NSTauM.hpp
  evaluators/PHAL_NSTauM_Def.hpp
  evaluators/PHAL_NSTauT.hpp
  evaluators/PHAL_NSTauT_Def.hpp
  evaluators/PHAL_NSThermalEqResid.hpp
  evaluators/PHAL_NSThermalEqResid_Def.hpp
  evaluators/PHAL_ODEResid.hpp
  evaluators/PHAL_ODEResid_Def.hpp
  evaluators/PNP_ConcentrationResid.hpp
  evaluators/PNP_ConcentrationResid_Def.hpp
  evaluators/PNP_PotentialResid.hpp
  evaluators/PNP_PotentialResid_Def.hpp
  )

# ASCR
IF(ALBANY_ASCR AND ALBANY_EPETRA)
  SET(SOURCES ${SOURCES}
    Albany_Networks.cpp
    )
  SET(HEADERS ${HEADERS}
    Albany_Networks.hpp
    )
ENDIF()

IF(ALBANY_HYDRIDE AND (NOT ALBANY_DEMO_PDES))
  SET(SOURCES ${SOURCES}
    evaluators/PHAL_LangevinNoiseTerm.cpp
    )
  SET(HEADERS ${HEADERS}
    evaluators/PHAL_LangevinNoiseTerm.hpp
    evaluators/PHAL_LangevinNoiseTerm_Def.hpp
    )
ENDIF()


SET(MESHMOVER_DIR ${Albany_SOURCE_DIR}/src/MeshMover/src)
SET(CUTR_SRCS
  ${MESHMOVER_DIR}/CUTR_CubitMeshMover.cpp
  ${MESHMOVER_DIR}/MeshMorpher.cpp
  ${MESHMOVER_DIR}/MeshMorpher_BMS.cpp
  ${MESHMOVER_DIR}/MeshMorpher_smooth.cpp
  ${MESHMOVER_DIR}/MeshMorpher_XWR.cpp
  ${MESHMOVER_DIR}/MoverProblem.cpp
  ${MESHMOVER_DIR}/STKMeshData.cpp
  ${MESHMOVER_DIR}/ToyProblem.cpp
  ${MESHMOVER_DIR}/util/CubitBox.cpp
  ${MESHMOVER_DIR}/util/CubitMatrix.cpp
  ${MESHMOVER_DIR}/util/CubitPlane.cpp
  ${MESHMOVER_DIR}/util/CubitVector.cpp
  ${MESHMOVER_DIR}/util/GeoNode.cpp
  ${MESHMOVER_DIR}/util/GeoTet.cpp
  ${MESHMOVER_DIR}/util/KDDTree.cpp
  ${MESHMOVER_DIR}/util/KDDTreeNode.cpp
  ${MESHMOVER_DIR}/util/MemoryBlock.cpp
  ${MESHMOVER_DIR}/util/MemoryManager.cpp
  ${MESHMOVER_DIR}/util/MMGfxDebug.cpp
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
  ${MESHMOVER_DIR}/MeshMorpher_BMS.hpp
  ${MESHMOVER_DIR}/MeshMorpher_smooth.hpp
  ${MESHMOVER_DIR}/MeshMorpher_XWR.hpp
  ${MESHMOVER_DIR}/MoverProblem.hpp
  ${MESHMOVER_DIR}/STKMeshData.hpp
  ${MESHMOVER_DIR}/ToyProblem.hpp
  ${MESHMOVER_DIR}/util/CubitBox.cpp
  ${MESHMOVER_DIR}/util/CubitMatrix.cpp
  ${MESHMOVER_DIR}/util/CubitPlane.cpp
  ${MESHMOVER_DIR}/util/CubitVector.cpp
  ${MESHMOVER_DIR}/util/GeoNode.cpp
  ${MESHMOVER_DIR}/util/GeoTet.cpp
  ${MESHMOVER_DIR}/util/KDDTree.cpp
  ${MESHMOVER_DIR}/util/KDDTreeNode.cpp
  ${MESHMOVER_DIR}/util/MemoryBlock.cpp
  ${MESHMOVER_DIR}/util/MemoryManager.cpp
  ${MESHMOVER_DIR}/util/MMGfxDebug.cpp
  ${MESHMOVER_DIR}/util/PriorityQueue.cpp
  ${MESHMOVER_DIR}/util/SimpleNode.cpp
  ${MESHMOVER_DIR}/util/TDDelaunay.cpp
  ${MESHMOVER_DIR}/util/TetFacetorTool.cpp
  ${MESHMOVER_DIR}/util/ToolData.cpp
  ${MESHMOVER_DIR}/util/ToolDataUser.cpp
  )

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

if (ALBANY_MOR AND ALBANY_EPETRA)
  include_directories (./MOR)

  set (MOR_ALBANY_HEADERS
    MOR/Albany_DiscretizationDofListProvider.hpp
    MOR/Albany_MORDiscretizationUtils.hpp
    MOR/Albany_MORFacade.hpp
    MOR/Albany_MORFacadeImpl.hpp
    MOR/Albany_StkEpetraMVSource.hpp
    MOR/Albany_StkNodalBasisSource.hpp
    )

  set (MOR_ALBANY_SOURCES
    MOR/Albany_DiscretizationDofListProvider.cpp
    MOR/Albany_MORDiscretizationUtils.cpp
    MOR/Albany_MORFacadeImpl.cpp
    MOR/Albany_StkEpetraMVSource.cpp
    MOR/Albany_StkNodalBasisSource.cpp
    )

  set(SOURCES ${SOURCES} ${MOR_ALBANY_SOURCES})
  set(HEADERS ${HEADERS} ${MOR_ALBANY_HEADERS})
endif ()

IF (ALBANY_SCOREC_STANDALONE)
  SET(SCOREC_LIB SCOREC::core)
ENDIF()

add_library(albanyLib ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})
target_link_libraries(albanyLib ${SCOREC_LIB} ${Trilinos_LIBRARIES})

# Add Albany external libraries

set(ALB_TRILINOS_LIBS ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})

include_directories (${Trilinos_INCLUDE_DIRS}
                     ${Trilinos_TPL_INCLUDE_DIRS}
                     ${ALBANY_PUMI_INCLUDE_DIRS}
                     ./
                     ./evaluators
                     ./problems
                     ./responses
                     ./disc/stk
                     ./adapt
                     ./disc
                     ./LCM/
                     ./LCM/evaluators
                     ./LCM/evaluators/bc
                     ./LCM/evaluators/contact
                     ./LCM/evaluators/HMC
                     ./LCM/evaluators/kinematics
                     ./LCM/evaluators/lame
                     ./LCM/evaluators/linear-elasticity
                     ./LCM/evaluators/peridigm
                     ./LCM/evaluators/poro
                     ./LCM/evaluators/projection
                     ./LCM/evaluators/responses
                     ./LCM/evaluators/residuals
                     ./LCM/evaluators/surface-element
                     ./LCM/evaluators/size-field
                     ./LCM/evaluators/transport
                     ./LCM/evaluators/HMC
                     ./LCM/models
                     ./LCM/utils
                     ./LCM/utils/topology
                     ./FELIX/problems
                     ./FELIX/evaluators
                     ./Aeras/problems
                     ./Aeras/evaluators
                     ./Hydride/problems
                     ./Hydride/evaluators
                     ./AMP/
                     ./AMP/problems
                     ./AMP/evaluators
                     ./AMP/responses
                     ./ANISO
                     ./QCAD
                     ./QCAD/problems
                     ./QCAD/evaluators
                     ./QCAD/responses
                     ./ATO
                     ./ATO/problems
                     ./ATO/evaluators
                     ./ATO/utils
                    )

link_directories (${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})

IF ( ENABLE_ISOLIB )
  include_directories (${ISOLIB_INCLUDE_DIR_MAIN})
  include_directories (${ISOLIB_INCLUDE_DIR_UTIL})
  link_directories (${ISOLIB_LIBRARY_DIR})
  set(ISOLIB_LIB iso)
ELSE()
  set(ISOLIB_LIB )
ENDIF()

IF ( ENABLE_NLopt )
  include_directories (${NLopt_INCLUDE_DIR})
  link_directories (${NLopt_LIBRARY_DIR})
  set(NLOPT_LIB nlopt)
ELSE()
  set(NLOPT_LIB )
ENDIF()

IF ( ENABLE_Cogent )
  include_directories (${Cogent_INCLUDE_DIR})
  link_directories (${Cogent_LIBRARY_DIR})
  set(Cogent_LIB Cogent)
ELSE()
  set(Cogent_LIB )
ENDIF()

IF ( ENABLE_DOTK )
  include_directories (${DOTK_INCLUDE_DIR})
  link_directories (${DOTK_LIBRARY_DIR})
  set(DOTK_LIB )
ELSE()
  set(DOTK_LIB )
ENDIF()


# LAME
IF(ALBANY_LAME)
  include_directories (${LAME_INCLUDE_DIR})
  link_directories (${LAME_LIBRARY_DIR})
  set(LAME_LIB lame ${LAME_LIBRARIES})
  include_directories (${MATH_TOOLKIT_INCLUDE_DIR})
  link_directories (${MATH_TOOLKIT_LIBRARY_DIR})
  set(MATH_TOOLKIT_LIB math_toolkit)
ELSE()
  set(LAME_LIB )
  set(MATH_TOOLKIT_LIB )
ENDIF()

# LAMENT
IF(ALBANY_LAMENT)
  include_directories (${LAMENT_INCLUDE_DIR})
  link_directories (${LAMENT_LIBRARY_DIR})
  set(LAMENT_LIB lament)
ELSE()
  set(LAMENT_LIB )
ENDIF()

# PERIDIGM
IF(ALBANY_PERIDIGM AND ALBANY_EPETRA)
  include_directories (${Peridigm_INCLUDE_DIRS})
  link_directories (${Peridigm_LIBRARY_DIRS})
ELSE()
  set(Peridigm_LIBRARIES )
ENDIF()

# Albany CI (Configuration Interaction)
IF(ALBANY_CI)
  include_directories (${ALBANY_CI_INCLUDE_DIR})
  link_directories (${ALBANY_CI_LIBRARY_DIR})
  set(ALBANY_CI_LIB albanyCILib)
ELSE()
  set(ALBANY_CI_LIB )
ENDIF()

# RPI APF mesh interface
IF(ALBANY_APF)
  SET(ALBANY_PUMI_INCLUDE_DIRS ./disc/pumi)
  include_directories (${ALBANY_PUMI_INCLUDE_DIRS})
ELSE()
  SET(ALBANY_PUMI_INCLUDE_DIRS )
ENDIF()

# Catalyst CoProcessing
if(ALBANY_CATALYST)
  set(ALBANY_CATALYST_INCLUDE_DIRS ./disc/catalyst)
  include_directories(${ALBANY_CATALYST_INCLUDE_DIRS})
endif()


# Now build executables from this library, tailored main()s, and Trilinos
IF (NOT ALBANY_LIBRARIES_ONLY)

IF (ALBANY_EPETRA)
  add_executable(Albany Main_Solve.cpp)
ENDIF()

add_executable(AlbanyT Main_SolveT.cpp)
IF (ALBANY_TEMPUS) 
  add_executable(AlbanyTempus Main_SolveT_Tempus.cpp) 
ENDIF() 

SET(ALBANY_EXECUTABLES AlbanyT)
IF (ALBANY_EPETRA)
  SET (ALBANY_EXECUTABLES ${ALBANY_EXECUTABLES} Albany )
ENDIF()
IF (ALBANY_TEMPUS) 
  SET (ALBANY_EXECUTABLES ${ALBANY_EXECUTABLES} AlbanyTempus )
ENDIF()

if (ALBANY_STOKHOS AND ALBANY_EPETRA)
add_executable(AlbanySG Main_SGSolve.cpp)
SET(ALBANY_EXECUTABLES ${ALBANY_EXECUTABLES} AlbanySG )
endif()
if (ALBANY_ASCR AND ALBANY_EPETRA)
  add_executable(AlbanyAdjoint Main_Adjoint.cpp)
  add_executable(AlbanySGAdjoint Main_SGAdjoint.cpp)
  add_executable(AlbanyCoupled Main_Coupled.cpp)
  add_executable(AlbanySGCoupled Main_SGCoupled.cpp)
SET(ALBANY_EXECUTABLES ${ALBANY_EXECUTABLES}
    AlbanyAdjoint
    AlbanyCoupled
    AlbanySGAdjoint
    AlbanySGCoupled
    )
endif()
IF (ALBANY_DAKOTA AND ALBANY_EPETRA)
  add_executable(AlbanyDakota Main_Dakota.cpp Albany_Dakota.cpp Albany_Dakota.hpp)
  SET(ALBANY_EXECUTABLES ${ALBANY_EXECUTABLES} AlbanyDakota)
ENDIF ()
IF (ALBANY_DAKOTA)
  add_executable(AlbanyDakotaT Main_DakotaT.cpp Albany_DakotaT.cpp Albany_DakotaT.hpp)
  SET(ALBANY_EXECUTABLES ${ALBANY_EXECUTABLES} AlbanyDakotaT)
ENDIF ()

IF (ALBANY_MOR AND ALBANY_EPETRA)
  IF (ALBANY_RBGEN)
    add_executable(AlbanyRBGen MOR/Main_RBGen.cpp)
    SET(ALBANY_EXECUTABLES ${ALBANY_EXECUTABLES} AlbanyRBGen)
  ENDIF ()
  IF (ALBANY_SEACAS)
    add_executable(AlbanyMeshSample MOR/Main_MeshSample.cpp)
    SET(ALBANY_EXECUTABLES ${ALBANY_EXECUTABLES} AlbanyMeshSample)
  ENDIF ()
  add_executable(AlbanyRomPostProcess MOR/Main_RomPostProcess.cpp)
  SET(ALBANY_EXECUTABLES ${ALBANY_EXECUTABLES} AlbanyRomPostProcess)
ENDIF ()

IF (ALBANY_EPETRA)
add_executable(AlbanyAnalysis Main_Analysis.cpp)
SET(ALBANY_EXECUTABLES ${ALBANY_EXECUTABLES} AlbanyAnalysis)
ENDIF()
add_executable(AlbanyAnalysisT Main_AnalysisT.cpp)
SET(ALBANY_EXECUTABLES ${ALBANY_EXECUTABLES} AlbanyAnalysisT)

IF (ALBANY_MESHDB_TOOLS)
  add_executable(exopumiconvert disc/tools/exopumiconvert.cpp)
  SET(ALBANY_EXECUTABLES ${ALBANY_EXECUTABLES} exopumiconvert)
ENDIF()

ENDIF (NOT ALBANY_LIBRARIES_ONLY)
# End declaration of executables

# Note: Albany_EXTRA_LIBRARIES is to allow users to specify
# libraries that one cannot get out of Trilinos directly.
set(ALL_LIBRARIES
  ${Peridigm_LIBRARIES}
  ${ALBANY_CI_LIB}
  ${ALB_TRILINOS_LIBS}
  ${ALB_CUBIT_LIBS}
  ${LAME_LIB}
  ${MATH_TOOLKIT_LIB}
  ${NLOPT_LIB}
  ${Cogent_LIB}
  ${DOTK_LIB}
  ${LAMENT_LIB}
  ${Trilinos_EXTRA_LD_FLAGS}
  ${Albany_EXTRA_LIBRARIES}
  ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}
  )

# Add Albany internal libraries/physics sets, as enabled.

add_subdirectory(adapt)
add_subdirectory(disc)

IF(ALBANY_HAVE_STK)
SET(ALBANY_LIBRARIES albanyLib albanyAdapt albanySTK)
ELSE()
SET(ALBANY_LIBRARIES albanyLib albanyAdapt)
ENDIF()

IF(ALBANY_DEMO_PDES)
  add_library(albanyPDEs ${Albany_LIBRARY_TYPE} ${PDE_SRCS} ${PDE_HDRS})
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} albanyPDEs)
ENDIF()

IF (ALBANY_MOR AND ALBANY_EPETRA)
  add_subdirectory(MOR)
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} MOR)
ENDIF()

IF (ALBANY_QCAD)
  add_subdirectory(QCAD)
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} QCAD)
ENDIF()

IF (ALBANY_FELIX)
  add_subdirectory(FELIX)
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} FELIX)

  IF (ENABLE_MPAS_INTERFACE)
    SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} mpasInterface)
  ENDIF()

  IF (ENABLE_CISM_INTERFACE)
    IF (ENABLE_CISM_EPETRA)
      SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} cismInterfaceE)
    ELSE()
      SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} cismInterface)
    ENDIF()
  ENDIF()
ENDIF()

IF (ALBANY_AERAS)
  add_subdirectory(Aeras)
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} Aeras)
ENDIF()

# RPI APF mesh interface
IF(ALBANY_APF)
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} albanyPUMI)
ENDIF()

IF (ALBANY_AMP)
  add_subdirectory(AMP)
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} AMP)
ENDIF()

IF (ALBANY_ANISO)
  add_subdirectory(ANISO)
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} ANISO)
ENDIF()

IF(ALBANY_CATALYST)
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} albanyCatalyst)
endif()

IF (ALBANY_ATO)
    SET(ALBANY_LCM ON)
    add_subdirectory(ATO)
    SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} ATO)
    # iso library uses stk
    SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} ${ISOLIB_LIB})
ENDIF()

IF (ALBANY_HYDRIDE)
  # Hydride requires LCM. However, LCM builds several excecutables. If shared
  # libraries are used, then all symbols must be defined. Hence
  # add_subdirectory(Hydride) must be called before, not after,
  # add_subdirectory(LCM) so that LCM/CMakeLists.txt knows about the Hydride
  # shared lib.
  add_subdirectory(Hydride)
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} hydride)
ENDIF()

IF (ALBANY_LCM)
  add_subdirectory(LCM)
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} LCM)
  add_executable(SchwarzAlternating Main_Alternating.cpp)
  SET (ALBANY_EXECUTABLES ${ALBANY_EXECUTABLES} SchwarzAlternating)
ENDIF()

# Can remove this once rebalance is in Trilnos again
IF(ALBANY_STK_REBALANCE)
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} albanySTKRebalance)
endif()

# Repeat libraries twice to catch circular dependencies
SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} ${ALBANY_LIBRARIES} ${ALBANY_LIBRARIES})

MESSAGE("-- Albany Executables to link: ${ALBANY_EXECUTABLES}")

IF (Albany_BUILD_STATIC_EXE)
  FOREACH(ALB_EXEC ${ALBANY_EXECUTABLES})
    SET_PROPERTY(TARGET ${ALB_EXEC} PROPERTY LINK_SEARCH_START_STATIC 1)
  ENDFOREACH()
ENDIF()

FOREACH(ALB_EXEC ${ALBANY_EXECUTABLES})
  target_link_libraries(${ALB_EXEC} ${ALBANY_LIBRARIES} ${ALL_LIBRARIES})
ENDFOREACH()

IF (INSTALL_ALBANY)
  configure_package_config_file(AlbanyConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/AlbanyConfig.cmake
    INSTALL_DESTINATION ${LIB_INSTALL_DIR}/Albany/cmake
    PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR)
  install(FILES ${HEADERS}
    DESTINATION ${INCLUDE_INSTALL_DIR})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/AlbanyConfig.cmake
    DESTINATION ${LIB_INSTALL_DIR}/Albany/cmake )
  # Install each binary in the list
  install(TARGETS ${ALBANY_EXECUTABLES} RUNTIME DESTINATION
     ${BINARY_INSTALL_DIR})
  install(TARGETS albanyLib EXPORT albany-export
    LIBRARY DESTINATION ${LIB_INSTALL_DIR}
    ARCHIVE DESTINATION ${LIB_INSTALL_DIR})
  IF (ENABLE_DEMO_PDES)
    install(TARGETS albanyPDEs EXPORT albany-export
      LIBRARY DESTINATION ${LIB_INSTALL_DIR}
      ARCHIVE DESTINATION ${LIB_INSTALL_DIR})
  ENDIF()
  INSTALL(EXPORT albany-export DESTINATION ${LIB_INSTALL_DIR}/Albany/cmake  FILE "albany-targets.cmake")

  SET(TMP1 ${ALBANY_LIBRARIES} ${ALL_LIBRARIES})

  MESSAGE("-- Exporting link libs to: ${CMAKE_INSTALL_PREFIX}/export_albany.in")
  STRING(REGEX REPLACE ";/" " /" TMP2 "${TMP1}")
  STRING(REPLACE ";" " -l" ALBANY_LINK_LIBS
        "-L${CMAKE_INSTALL_PREFIX}/${LIB_INSTALL_DIR}"
        " -L${Trilinos_LIBRARY_DIRS}" " -l${TMP2}")
  CONFIGURE_FILE(export_albany.in ${CMAKE_INSTALL_PREFIX}/export_albany.in)

ENDIF()
