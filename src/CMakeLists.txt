##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Configure the file containing Albany configuration macros,
# and add its folder to the include directories
CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/Albany_config.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/Albany_config.h)
INCLUDE_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR})


INCLUDE(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${Trilinos_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LIBRARIES ${Trilinos_LIBRARY_DIRS})
set( CMAKE_REQUIRED_FLAGS ${Trilinos_CXX_FLAGS} )


SET(SOURCES
  Albany_SolverFactory.cpp
  Albany_Utils.cpp
  PHAL_Dimension.cpp
  PHAL_Setup.cpp
  Albany_Application.cpp
  Albany_Memory.cpp
  Albany_ModelEvaluator.cpp
  Albany_NullSpaceUtils.cpp
  Albany_ObserverImpl.cpp
  Albany_PiroObserver.cpp
  Albany_StatelessObserverImpl.cpp
  Albany_StateManager.cpp
  Albany_StateInfoStruct.cpp
  PHAL_Utilities.cpp
  )

IF (ALBANY_EPETRA)
  SET(SOURCES ${SOURCES} Albany_ObserverFactory.cpp)
  SET(SOURCES ${SOURCES} Petra_Converters_64.cpp)
ENDIF()


IF(ALBANY_EPETRA)
  SET(SOURCES ${SOURCES}
    Albany_NOXObserver.cpp
    Albany_NOXStatelessObserver.cpp
  )
  IF(ALBANY_RYTHMOS)
    SET(SOURCES ${SOURCES} Albany_RythmosObserver.cpp)
  ENDIF()
ENDIF()

SET(HEADERS
  Albany_Application.hpp
  Albany_DataTypes.hpp
  Albany_DistributedParameter.hpp
  Albany_DistributedParameterLibrary.hpp
  Albany_DistributedParameterDerivativeOp.hpp
  Albany_DummyParameterAccessor.hpp
  Albany_EigendataInfoStructT.hpp
  Albany_KokkosTypes.hpp
  Albany_Memory.hpp
  Albany_ModelEvaluator.hpp
  Albany_NullSpaceUtils.hpp
  Albany_ObserverImpl.hpp
  Albany_PiroObserver.hpp
  Albany_ScalarOrdinalTypes.hpp
  Albany_SolverFactory.hpp
  Albany_StateManager.hpp
  Albany_StateInfoStruct.hpp
  Albany_StatelessObserverImpl.hpp
  Albany_TpetraThyraTypes.hpp
  Albany_TpetraTypes.hpp
  Albany_ThyraTypes.hpp
  Albany_Utils.hpp
  PHAL_AlbanyTraits.hpp
  PHAL_Dimension.hpp
  PHAL_FactoryTraits.hpp
  PHAL_Setup.hpp
  PHAL_TypeKeyMap.hpp
  PHAL_Utilities.hpp
  PHAL_Utilities_Def.hpp
  PHAL_Workset.hpp
  )

IF(ALBANY_EPETRA)
  SET(HEADERS ${HEADERS}
    Albany_EigendataInfoStruct.hpp
    Albany_NOXObserver.hpp
    Albany_NOXStatelessObserver.hpp
    Albany_ObserverFactory.hpp
    Petra_Converters.hpp
  )
  IF(ALBANY_RYTHMOS)
    SET(SOURCES ${SOURCES} Albany_RythmosObserver.hpp)
  ENDIF()
ENDIF()

#utility
SET(SOURCES ${SOURCES}
  utility/Counter.cpp
  utility/CounterMonitor.cpp
  utility/DisplayTable.cpp
  utility/PerformanceContext.cpp
  utility/TimeMonitor.cpp
  utility/Albany_CombineAndScatterManager.cpp
  utility/Albany_CombineAndScatterManagerTpetra.cpp
  utility/Albany_CommUtils.cpp
  utility/Albany_Gather.cpp
  utility/Albany_GlobalLocalIndexer.cpp
  utility/Albany_ThyraCrsMatrixFactory.cpp
  utility/Albany_ThyraUtils.cpp
  utility/Albany_TpetraThyraUtils.cpp
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
  utility/Albany_CombineAndScatterManager.hpp
  utility/Albany_CombineAndScatterManagerTpetra.hpp
  utility/Albany_CommUtils.hpp
  utility/Albany_Gather.hpp
  utility/Albany_GlobalLocalIndexer.hpp
  utility/Albany_GlobalLocalIndexerTpetra.hpp
  utility/Albany_ThyraCrsMatrixFactory.hpp
  utility/Albany_ThyraUtils.hpp
  utility/Albany_TpetraThyraUtils.hpp
  utility/VariableMonitor.hpp
  utility/StaticAllocator.hpp
  utility/math/Tensor.hpp
  utility/math/TensorCommon.hpp
  utility/math/TensorDetail.hpp
  utility/math/TensorImpl.hpp
  )

IF (ALBANY_EPETRA)
  SET(SOURCES ${SOURCES}
    utility/Albany_EpetraThyraUtils.cpp
    utility/Albany_CombineAndScatterManagerEpetra.cpp
  )
  SET(HEADERS ${HEADERS}
    utility/Albany_EpetraThyraUtils.hpp
    utility/Albany_CombineAndScatterManagerEpetra.hpp
    utility/Albany_GlobalLocalIndexerEpetra.hpp
  )
ENDIF()

add_executable(xml2yaml utility/xml2yaml.cpp)
add_executable(yaml2xml utility/yaml2xml.cpp)
target_link_libraries(xml2yaml teuchosparameterlist)
target_link_libraries(yaml2xml teuchosparameterlist)

#problems
SET(SOURCES ${SOURCES}
  problems/Albany_AbstractProblem.cpp
  problems/Albany_BCUtils.cpp
  problems/Albany_EvaluatorUtils.cpp
  problems/Albany_HeatProblem.cpp
  problems/Albany_ThermalProblem.cpp
  problems/Albany_Layouts.cpp
  problems/Albany_PopulateMesh.cpp
  problems/Albany_ProblemFactory.cpp
  problems/Albany_ProblemUtils.cpp
  problems/Albany_ResponseUtilities.cpp
  problems/Albany_SideLaplacianProblem.cpp
  problems/Albany_MaterialDatabase.cpp
  )
SET(PDE_SRCS
  problems/Albany_AdvDiffProblem.cpp
  problems/Albany_ReactDiffSystem.cpp
  problems/Albany_CahnHillProblem.cpp
  problems/Albany_ComprNSProblem.cpp
  problems/Albany_Helmholtz2DProblem.cpp
  problems/Albany_LinComprNSProblem.cpp
  problems/Albany_ODEProblem.cpp
  problems/Albany_NavierStokes.cpp
  problems/Albany_PNPProblem.cpp
  problems/Albany_ThermoElectrostaticsProblem.cpp
  )
SET(HEADERS ${HEADERS}
  problems/Albany_AbstractProblem.hpp
  problems/Albany_BCUtils.hpp
  problems/Albany_BCUtils_Def.hpp
  problems/Albany_EvaluatorUtils.hpp
  problems/Albany_EvaluatorUtils_Def.hpp
  problems/Albany_GeneralPurposeFieldsNames.hpp
  problems/Albany_HeatProblem.hpp
  problems/Albany_ThermalProblem.hpp
  problems/Albany_Layouts.hpp
  problems/Albany_PopulateMesh.hpp
  problems/Albany_ProblemFactory.hpp
  problems/Albany_ProblemUtils.hpp
  problems/Albany_ResponseUtilities.hpp
  problems/Albany_ResponseUtilities_Def.hpp
  problems/Albany_SideLaplacianProblem.hpp
  problems/Albany_MaterialDatabase.hpp
  )
SET(PDE_HDRS
  problems/Albany_AdvDiffProblem.hpp
  problems/Albany_ReactDiffSystem.hpp
  problems/Albany_CahnHillProblem.hpp
  problems/Albany_ComprNSProblem.hpp
  problems/Albany_Helmholtz2DProblem.hpp
  problems/Albany_LinComprNSProblem.hpp
  problems/Albany_NavierStokes.hpp
  problems/Albany_ODEProblem.hpp
  problems/Albany_PNPProblem.hpp
  problems/Albany_ThermoElectrostaticsProblem.hpp
  )

#responses
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
  responses/Albany_SolutionResponseFunction.cpp
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
  responses/Albany_SolutionMinValueResponseFunction.cpp
  responses/Albany_SolutionResponseFunction.hpp
  responses/Albany_SolutionTwoNormResponseFunction.hpp
  responses/Albany_SolutionValuesResponseFunction.hpp
  )

#discretization
SET(SOURCES ${SOURCES}
  disc/Adapt_NodalDataBase.cpp
  disc/Adapt_NodalDataVector.cpp
  disc/Albany_DiscretizationFactory.cpp
  disc/Albany_MeshSpecs.cpp
  )
SET(HEADERS ${HEADERS}
  disc/Adapt_NodalDataBase.hpp
  disc/Adapt_NodalDataVector.hpp
  disc/Adapt_NodalFieldUtils.hpp
  disc/Albany_DiscretizationUtils.hpp
  disc/Albany_AbstractDiscretization.hpp
  disc/Albany_AbstractFieldContainer.hpp
  disc/Albany_AbstractMeshStruct.hpp
  disc/Albany_AbstractNodeFieldContainer.hpp
  disc/Albany_DiscretizationFactory.hpp
  disc/Albany_MeshSpecs.hpp
  disc/Albany_NodalDOFManager.hpp
  )

IF (ALBANY_CONTACT)

  SET(SOURCES ${SOURCES}
    disc/Albany_ContactManager.cpp
    evaluators/scatter/PHAL_MortarContactResidual.cpp
  )
  SET(HEADERS ${HEADERS}
    disc/Albany_ContactManager.hpp
    evaluators/scatter/PHAL_MortarContactResidual.hpp
    evaluators/scatter/PHAL_MortarContactResidual_Def.hpp
  )
ENDIF ()

#evaluators
SET(SOURCES ${SOURCES}
  evaluators/bc/PHAL_Dirichlet.cpp
  evaluators/bc/PHAL_TimeDepDBC.cpp
  evaluators/bc/PHAL_TimeDepSDBC.cpp
  evaluators/bc/PHAL_SDirichlet.cpp
  evaluators/bc/PHAL_DirichletCoordinateFunction.cpp
  evaluators/bc/PHAL_DirichletField.cpp
  evaluators/bc/PHAL_SDirichletField.cpp
  evaluators/bc/PHAL_DirichletOffNodeSet.cpp
  evaluators/bc/PHAL_IdentityCoordinateFunctionTraits.cpp
  evaluators/bc/PHAL_Neumann.cpp
  evaluators/gather/PHAL_GatherAuxData.cpp
  evaluators/gather/PHAL_GatherCoordinateVector.cpp
  evaluators/gather/PHAL_GatherScalarNodalParameter.cpp
  evaluators/gather/PHAL_GatherSolution.cpp
  evaluators/interpolation/PHAL_DOFCellToSide.cpp
  evaluators/interpolation/PHAL_DOFCellToSideQP.cpp
  evaluators/interpolation/PHAL_DOFGradInterpolation.cpp
  evaluators/interpolation/PHAL_DOFGradInterpolationSide.cpp
  evaluators/interpolation/PHAL_DOFInterpolation.cpp
  evaluators/interpolation/PHAL_DOFInterpolationSide.cpp
  evaluators/interpolation/PHAL_DOFSideToCell.cpp
  evaluators/interpolation/PHAL_DOFTensorGradInterpolation.cpp
  evaluators/interpolation/PHAL_DOFTensorInterpolation.cpp
  evaluators/interpolation/PHAL_DOFVecGradInterpolation.cpp
  evaluators/interpolation/PHAL_DOFVecGradInterpolationSide.cpp
  evaluators/interpolation/PHAL_DOFVecInterpolation.cpp
  evaluators/interpolation/PHAL_DOFVecInterpolationSide.cpp
  evaluators/interpolation/PHAL_NodesToCellInterpolation.cpp
  evaluators/interpolation/PHAL_QuadPointsToCellInterpolation.cpp
  evaluators/interpolation/PHAL_SideQuadPointsToSideInterpolation.cpp
  evaluators/pde/PHAL_HeatEqResid.cpp
  evaluators/pde/PHAL_ThermalResid.cpp
  evaluators/pde/PHAL_NSMaterialProperty.cpp
  evaluators/pde/PHAL_SideLaplacianResidual.cpp
  evaluators/pde/PHAL_ThermalConductivity.cpp
  evaluators/response/PHAL_ResponseFieldIntegral.cpp
  evaluators/response/PHAL_ResponseSquaredL2Difference.cpp
  evaluators/response/PHAL_ResponseSquaredL2DifferenceSide.cpp
  evaluators/response/PHAL_ResponseThermalEnergy.cpp
  evaluators/response/QCAD_ResponseCenterOfMass.cpp
  evaluators/response/QCAD_ResponseFieldAverage.cpp
  evaluators/response/QCAD_ResponseFieldIntegral.cpp
  evaluators/response/QCAD_ResponseFieldValue.cpp
  evaluators/scatter/PHAL_ScatterResidual.cpp
  evaluators/scatter/PHAL_ScatterScalarNodalParameter.cpp
  evaluators/scatter/PHAL_ScatterScalarResponse.cpp
  evaluators/scatter/PHAL_SeparableScatterScalarResponse.cpp
  evaluators/state/PHAL_LoadSideSetStateField.cpp
  evaluators/state/PHAL_LoadStateField.cpp
  evaluators/state/PHAL_ReadStateField.cpp
  evaluators/state/PHAL_SaveCellStateField.cpp
  evaluators/state/PHAL_SaveNodalField.cpp
  evaluators/state/PHAL_SaveStateField.cpp
  evaluators/state/QCAD_ResponseSaveField.cpp
  evaluators/utility/Adapt_ElementSizeField.cpp
  evaluators/utility/PHAL_Absorption.cpp
  evaluators/utility/PHAL_AddNoise.cpp
  evaluators/utility/PHAL_ComputeBasisFunctions.cpp
  evaluators/utility/PHAL_ComputeBasisFunctionsSide.cpp
  evaluators/utility/PHAL_Constant.cpp
  evaluators/utility/PHAL_ConvertFieldType.cpp
  evaluators/utility/PHAL_DummyResidual.cpp
  evaluators/utility/PHAL_FieldFrobeniusNorm.cpp
  evaluators/utility/PHAL_LangevinNoiseTerm.cpp
  evaluators/utility/PHAL_MapToPhysicalFrame.cpp
  evaluators/utility/PHAL_MapToPhysicalFrameSide.cpp
  evaluators/utility/PHAL_SharedParameter.cpp
  evaluators/utility/PHAL_Source.cpp
  evaluators/utility/QCAD_EvaluatorTools.cpp
  evaluators/utility/QCAD_MathVector.cpp
  evaluators/utility/QCAD_MeshRegion.cpp
  )

IF (ALBANY_STK)
 SET(SOURCES ${SOURCES}
   evaluators/state/PHAL_SaveSideSetStateField.cpp)
  IF(ALBANY_STK_EXPR_EVAL)
    SET(SOURCES ${SOURCES} 
      evaluators/bc/PHAL_ExprEvalSDBC.cpp)
  ENDIF()
ENDIF()

IF (ALBANY_EPETRA)
  SET(SOURCES ${SOURCES}
    evaluators/gather/PHAL_GatherEigenData.cpp
    evaluators/gather/PHAL_GatherEigenvectors.cpp
  )
ENDIF()

SET(PDE_SRCS ${PDE_SRCS}
  evaluators/pde/PHAL_AdvDiffResid.cpp
  evaluators/pde/PHAL_ReactDiffSystemResid.cpp
  evaluators/pde/PHAL_CahnHillChemTerm.cpp
  evaluators/pde/PHAL_CahnHillRhoResid.cpp
  evaluators/pde/PHAL_CahnHillWResid.cpp
  evaluators/pde/PHAL_ComprNSBodyForce.cpp
  evaluators/pde/PHAL_ComprNSResid.cpp
  evaluators/pde/PHAL_ComprNSViscosity.cpp
  evaluators/pde/PHAL_HelmholtzResid.cpp
  evaluators/pde/PHAL_JouleHeating.cpp
  evaluators/pde/PHAL_LinComprNSBodyForce.cpp
  evaluators/pde/PHAL_LinComprNSResid.cpp
  evaluators/pde/PHAL_NSBodyForce.cpp
  evaluators/pde/PHAL_NSContinuityResid.cpp
  evaluators/pde/PHAL_NSContravarientMetricTensor.cpp
  evaluators/pde/PHAL_NSForchheimerTerm.cpp
  evaluators/pde/PHAL_NSMomentumResid.cpp
  evaluators/pde/PHAL_NSNeutronEqResid.cpp
  evaluators/pde/PHAL_NSPermeabilityTerm.cpp
  evaluators/pde/PHAL_NSRm.cpp
  evaluators/pde/PHAL_NSTauM.cpp
  evaluators/pde/PHAL_NSTauT.cpp
  evaluators/pde/PHAL_NSThermalEqResid.cpp
  evaluators/pde/PHAL_ODEResid.cpp
  evaluators/pde/PHAL_PoissonResid.cpp
  evaluators/pde/PHAL_Permittivity.cpp
  evaluators/pde/PHAL_TEProp.cpp
  evaluators/pde/PNP_ConcentrationResid.cpp
  evaluators/pde/PNP_PotentialResid.cpp
  )

SET(HEADERS ${HEADERS}
  evaluators/bc/PHAL_Dirichlet.hpp
  evaluators/bc/PHAL_TimeDepDBC.hpp
  evaluators/bc/PHAL_TimeDepSDBC.hpp
  evaluators/bc/PHAL_SDirichlet.hpp
  evaluators/bc/PHAL_DirichletCoordinateFunction.hpp
  evaluators/bc/PHAL_DirichletCoordinateFunction_Def.hpp
  evaluators/bc/PHAL_DirichletField.hpp
  evaluators/bc/PHAL_DirichletField_Def.hpp
  evaluators/bc/PHAL_SDirichletField.hpp
  evaluators/bc/PHAL_SDirichletField_Def.hpp
  evaluators/bc/PHAL_DirichletOffNodeSet.hpp
  evaluators/bc/PHAL_DirichletOffNodeSet_Def.hpp
  evaluators/bc/PHAL_Dirichlet_Def.hpp
  evaluators/bc/PHAL_TimeDepDBC_Def.hpp
  evaluators/bc/PHAL_TimeDepSDBC_Def.hpp
  evaluators/bc/PHAL_SDirichlet_Def.hpp
  evaluators/bc/PHAL_IdentityCoordinateFunctionTraits.hpp
  evaluators/bc/PHAL_IdentityCoordinateFunctionTraits_Def.hpp
  evaluators/bc/PHAL_Neumann.hpp
  evaluators/bc/PHAL_Neumann_Def.hpp
  evaluators/gather/PHAL_GatherAuxData.hpp
  evaluators/gather/PHAL_GatherAuxData_Def.hpp
  evaluators/gather/PHAL_GatherCoordinateVector.hpp
  evaluators/gather/PHAL_GatherCoordinateVector_Def.hpp
  evaluators/gather/PHAL_GatherScalarNodalParameter.hpp
  evaluators/gather/PHAL_GatherScalarNodalParameter_Def.hpp
  evaluators/gather/PHAL_GatherSolution.hpp
  evaluators/gather/PHAL_GatherSolution_Def.hpp
  evaluators/interpolation/PHAL_DOFCellToSide.hpp
  evaluators/interpolation/PHAL_DOFCellToSideQP.hpp
  evaluators/interpolation/PHAL_DOFCellToSideQP_Def.hpp
  evaluators/interpolation/PHAL_DOFCellToSide_Def.hpp
  evaluators/interpolation/PHAL_DOFGradInterpolation.hpp
  evaluators/interpolation/PHAL_DOFGradInterpolationSide.hpp
  evaluators/interpolation/PHAL_DOFGradInterpolationSide_Def.hpp
  evaluators/interpolation/PHAL_DOFGradInterpolation_Def.hpp
  evaluators/interpolation/PHAL_DOFInterpolation.hpp
  evaluators/interpolation/PHAL_DOFInterpolationSide.hpp
  evaluators/interpolation/PHAL_DOFInterpolationSide_Def.hpp
  evaluators/interpolation/PHAL_DOFInterpolation_Def.hpp
  evaluators/interpolation/PHAL_DOFSideToCell.hpp
  evaluators/interpolation/PHAL_DOFSideToCell_Def.hpp
  evaluators/interpolation/PHAL_DOFTensorGradInterpolation.hpp
  evaluators/interpolation/PHAL_DOFTensorGradInterpolation_Def.hpp
  evaluators/interpolation/PHAL_DOFTensorInterpolation.hpp
  evaluators/interpolation/PHAL_DOFTensorInterpolation_Def.hpp
  evaluators/interpolation/PHAL_DOFVecGradInterpolation.hpp
  evaluators/interpolation/PHAL_DOFVecGradInterpolationSide.hpp
  evaluators/interpolation/PHAL_DOFVecGradInterpolationSide_Def.hpp
  evaluators/interpolation/PHAL_DOFVecGradInterpolation_Def.hpp
  evaluators/interpolation/PHAL_DOFVecInterpolation.hpp
  evaluators/interpolation/PHAL_DOFVecInterpolationSide.hpp
  evaluators/interpolation/PHAL_DOFVecInterpolationSide_Def.hpp
  evaluators/interpolation/PHAL_DOFVecInterpolation_Def.hpp
  evaluators/interpolation/PHAL_NodesToCellInterpolation.hpp
  evaluators/interpolation/PHAL_NodesToCellInterpolation_Def.hpp
  evaluators/interpolation/PHAL_QuadPointsToCellInterpolation.hpp
  evaluators/interpolation/PHAL_QuadPointsToCellInterpolation_Def.hpp
  evaluators/interpolation/PHAL_SideQuadPointsToSideInterpolation.hpp
  evaluators/interpolation/PHAL_SideQuadPointsToSideInterpolation_Def.hpp
  evaluators/pde/PHAL_HeatEqResid.hpp
  evaluators/pde/PHAL_HeatEqResid_Def.hpp
  evaluators/pde/PHAL_ThermalResid.hpp
  evaluators/pde/PHAL_ThermalResid_Def.hpp
  evaluators/pde/PHAL_NSMaterialProperty.hpp
  evaluators/pde/PHAL_NSMaterialProperty_Def.hpp
  evaluators/pde/PHAL_SideLaplacianResidual.hpp
  evaluators/pde/PHAL_SideLaplacianResidual_Def.hpp
  evaluators/pde/PHAL_ThermalConductivity.hpp
  evaluators/pde/PHAL_ThermalConductivity_Def.hpp
  evaluators/response/PHAL_ResponseFieldIntegral.hpp
  evaluators/response/PHAL_ResponseFieldIntegral_Def.hpp
  evaluators/response/PHAL_ResponseSquaredL2Difference.hpp
  evaluators/response/PHAL_ResponseSquaredL2DifferenceSide.hpp
  evaluators/response/PHAL_ResponseSquaredL2DifferenceSide_Def.hpp
  evaluators/response/PHAL_ResponseSquaredL2Difference_Def.hpp
  evaluators/response/PHAL_ResponseThermalEnergy.hpp
  evaluators/response/PHAL_ResponseThermalEnergy_Def.hpp
  evaluators/response/QCAD_ResponseCenterOfMass.hpp
  evaluators/response/QCAD_ResponseCenterOfMass_Def.hpp
  evaluators/response/QCAD_ResponseFieldAverage.hpp
  evaluators/response/QCAD_ResponseFieldAverage_Def.hpp
  evaluators/response/QCAD_ResponseFieldIntegral.hpp
  evaluators/response/QCAD_ResponseFieldIntegral_Def.hpp
  evaluators/response/QCAD_ResponseFieldValue.hpp
  evaluators/response/QCAD_ResponseFieldValue_Def.hpp
  evaluators/scatter/PHAL_ScatterResidual.hpp
  evaluators/scatter/PHAL_ScatterResidual_Def.hpp
  evaluators/scatter/PHAL_ScatterScalarNodalParameter.hpp
  evaluators/scatter/PHAL_ScatterScalarNodalParameter_Def.hpp
  evaluators/scatter/PHAL_ScatterScalarResponse.hpp
  evaluators/scatter/PHAL_ScatterScalarResponse_Def.hpp
  evaluators/state/PHAL_LoadSideSetStateField.hpp
  evaluators/state/PHAL_LoadSideSetStateField_Def.hpp
  evaluators/state/PHAL_LoadStateField.hpp
  evaluators/state/PHAL_LoadStateField_Def.hpp
  evaluators/state/PHAL_ReadStateField.hpp
  evaluators/state/PHAL_ReadStateField_Def.hpp
  evaluators/state/PHAL_SaveCellStateField.hpp
  evaluators/state/PHAL_SaveCellStateField_Def.hpp
  evaluators/state/PHAL_SaveNodalField.hpp
  evaluators/state/PHAL_SaveNodalField_Def.hpp
  evaluators/state/PHAL_SaveStateField.hpp
  evaluators/state/PHAL_SaveStateField_Def.hpp
  evaluators/state/QCAD_ResponseSaveField.hpp
  evaluators/state/QCAD_ResponseSaveField_Def.hpp
  evaluators/utility/Adapt_ElementSizeField.hpp
  evaluators/utility/Adapt_ElementSizeField_Def.hpp
  evaluators/utility/PHAL_Absorption.hpp
  evaluators/utility/PHAL_Absorption_Def.hpp
  evaluators/utility/PHAL_AddNoise.hpp
  evaluators/utility/PHAL_AddNoise_Def.hpp
  evaluators/utility/PHAL_ComputeBasisFunctions.hpp
  evaluators/utility/PHAL_ComputeBasisFunctions_Def.hpp
  evaluators/utility/PHAL_ComputeBasisFunctionsSide.hpp
  evaluators/utility/PHAL_ComputeBasisFunctionsSide_Def.hpp
  evaluators/utility/PHAL_Constant.hpp
  evaluators/utility/PHAL_Constant_Def.hpp
  evaluators/utility/PHAL_ConvertFieldType.hpp
  evaluators/utility/PHAL_ConvertFieldType_Def.hpp
  evaluators/utility/PHAL_DummyResidual.hpp
  evaluators/utility/PHAL_DummyResidual_Def.hpp
  evaluators/utility/PHAL_FieldFrobeniusNorm.hpp
  evaluators/utility/PHAL_FieldFrobeniusNorm_Def.hpp
  evaluators/utility/PHAL_LangevinNoiseTerm.hpp
  evaluators/utility/PHAL_LangevinNoiseTerm_Def.hpp
  evaluators/utility/PHAL_MapToPhysicalFrame.hpp
  evaluators/utility/PHAL_MapToPhysicalFrameSide.hpp
  evaluators/utility/PHAL_MapToPhysicalFrameSide_Def.hpp
  evaluators/utility/PHAL_MapToPhysicalFrame_Def.hpp
  evaluators/utility/PHAL_SharedParameter.hpp
  evaluators/utility/PHAL_SharedParameter_Def.hpp
  evaluators/utility/PHAL_Source.hpp
  evaluators/utility/PHAL_Source_Def.hpp
  evaluators/utility/QCAD_EvaluatorTools.hpp
  evaluators/utility/QCAD_EvaluatorTools_Def.hpp
  evaluators/utility/QCAD_MathVector.hpp
  evaluators/utility/QCAD_MeshRegion.hpp
  evaluators/utility/QCAD_MeshRegion_Def.hpp
  )
  

IF (ALBANY_STK)
  SET(HEADERS ${HEADERS}
    evaluators/state/PHAL_SaveSideSetStateField.hpp
    evaluators/state/PHAL_SaveSideSetStateField_Def.hpp)
  IF(ALBANY_STK_EXPR_EVAL)
    SET(HEADERS ${HEADERS} 
      evaluators/bc/PHAL_ExprEvalSDBC.hpp
      evaluators/bc/PHAL_ExprEvalSDBC_Def.hpp)
  ENDIF()
ENDIF()


IF (ALBANY_EPETRA)
  SET(HEADERS ${HEADERS}
    evaluators/gather/PHAL_GatherEigenvectors.hpp
    evaluators/gather/PHAL_GatherEigenvectors_Def.hpp
    evaluators/gather/PHAL_GatherEigenData.hpp
    evaluators/gather/PHAL_GatherEigenData_Def.hpp
    evaluators/scatter/PHAL_SeparableScatterScalarResponse.hpp
    evaluators/scatter/PHAL_SeparableScatterScalarResponse_Def.hpp
  )
ENDIF()

SET(PDE_HDRS ${PDE_HDRS}
  evaluators/pde/PHAL_AdvDiffResid.hpp
  evaluators/pde/PHAL_AdvDiffResid_Def.hpp
  evaluators/pde/PHAL_ReactDiffSystemResid.hpp
  evaluators/pde/PHAL_ReactDiffSystemResid_Def.hpp
  evaluators/pde/PHAL_CahnHillRhoResid.hpp
  evaluators/pde/PHAL_CahnHillRhoResid_Def.hpp
  evaluators/pde/PHAL_CahnHillWResid.hpp
  evaluators/pde/PHAL_CahnHillWResid_Def.hpp
  evaluators/pde/PHAL_CahnHillChemTerm.hpp
  evaluators/pde/PHAL_CahnHillChemTerm_Def.hpp
  evaluators/pde/PHAL_ComprNSBodyForce.hpp
  evaluators/pde/PHAL_ComprNSBodyForce_Def.hpp
  evaluators/pde/PHAL_ComprNSResid.hpp
  evaluators/pde/PHAL_ComprNSResid_Def.hpp
  evaluators/pde/PHAL_ComprNSViscosity.hpp
  evaluators/pde/PHAL_ComprNSViscosity_Def.hpp
  evaluators/pde/PHAL_HelmholtzResid.hpp
  evaluators/pde/PHAL_HelmholtzResid_Def.hpp
  evaluators/pde/PHAL_JouleHeating.hpp
  evaluators/pde/PHAL_JouleHeating_Def.hpp
  evaluators/pde/PHAL_LinComprNSBodyForce.hpp
  evaluators/pde/PHAL_LinComprNSBodyForce_Def.hpp
  evaluators/pde/PHAL_LinComprNSResid.hpp
  evaluators/pde/PHAL_LinComprNSResid_Def.hpp
  evaluators/pde/PHAL_NSContinuityResid.hpp
  evaluators/pde/PHAL_NSContinuityResid_Def.hpp
  evaluators/pde/PHAL_NSBodyForce.hpp
  evaluators/pde/PHAL_NSBodyForce_Def.hpp
  evaluators/pde/PHAL_NSContravarientMetricTensor.hpp
  evaluators/pde/PHAL_NSContravarientMetricTensor_Def.hpp
  evaluators/pde/PHAL_NSForchheimerTerm.hpp
  evaluators/pde/PHAL_NSForchheimerTerm_Def.hpp
  evaluators/pde/PHAL_NSMomentumResid.hpp
  evaluators/pde/PHAL_NSMomentumResid_Def.hpp
  evaluators/pde/PHAL_NSNeutronEqResid.hpp
  evaluators/pde/PHAL_NSNeutronEqResid_Def.hpp
  evaluators/pde/PHAL_NSPermeabilityTerm.hpp
  evaluators/pde/PHAL_NSPermeabilityTerm_Def.hpp
  evaluators/pde/PHAL_NSRm.hpp
  evaluators/pde/PHAL_NSRm_Def.hpp
  evaluators/pde/PHAL_NSTauM.hpp
  evaluators/pde/PHAL_NSTauM_Def.hpp
  evaluators/pde/PHAL_NSTauT.hpp
  evaluators/pde/PHAL_NSTauT_Def.hpp
  evaluators/pde/PHAL_NSThermalEqResid.hpp
  evaluators/pde/PHAL_NSThermalEqResid_Def.hpp
  evaluators/pde/PHAL_ODEResid.hpp
  evaluators/pde/PHAL_ODEResid_Def.hpp
  evaluators/pde/PHAL_PoissonResid.hpp
  evaluators/pde/PHAL_PoissonResid_Def.hpp
  evaluators/pde/PHAL_Permittivity.hpp
  evaluators/pde/PHAL_Permittivity_Def.hpp
  evaluators/pde/PHAL_TEProp.hpp
  evaluators/pde/PHAL_TEProp_Def.hpp
  evaluators/pde/PNP_ConcentrationResid.hpp
  evaluators/pde/PNP_ConcentrationResid_Def.hpp
  evaluators/pde/PNP_PotentialResid.hpp
  evaluators/pde/PNP_PotentialResid_Def.hpp
  )

IF (ALBANY_SCOREC_STANDALONE)
  SET(SCOREC_LIB SCOREC::core)
ENDIF()

add_library(albanyLib ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})
target_link_libraries(albanyLib ${SCOREC_LIB} ${Trilinos_LIBRARIES})

# Add Albany external libraries

set(ALB_TRILINOS_LIBS ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})

# We may not want warnings from Trilinos and/or its TPLs...
IF (ALBANY_SUPPRESS_TRILINOS_WARNINGS)
  include_directories (SYSTEM
                       ${Trilinos_INCLUDE_DIRS}
                       ${Trilinos_TPL_INCLUDE_DIRS})
ELSE()
  include_directories (${Trilinos_INCLUDE_DIRS}
                       ${Trilinos_TPL_INCLUDE_DIRS})
ENDIF()

include_directories (${ALBANY_PUMI_INCLUDE_DIRS}
                     ./
                     ./evaluators
                     ./evaluators/bc
                     ./evaluators/gather
                     ./evaluators/interpolation
                     ./evaluators/pde
                     ./evaluators/response
                     ./evaluators/scatter
                     ./evaluators/state
                     ./evaluators/utility
                     ./problems
                     ./responses
                     ./disc/stk
                     ./adapt
                     ./disc
                     ./utility
                     ./LCM/
                     ./LCM/evaluators
                     ./LCM/evaluators/ACE
                     ./LCM/evaluators/bc
                     ./LCM/evaluators/contact
                     ./LCM/evaluators/HMC
                     ./LCM/evaluators/kinematics
                     ./LCM/evaluators/lame
                     ./LCM/evaluators/linear-elasticity
                     ./LCM/evaluators/poro
                     ./LCM/evaluators/projection
                     ./LCM/evaluators/responses
                     ./LCM/evaluators/residuals
                     ./LCM/evaluators/surface-element
                     ./LCM/evaluators/size-field
                     ./LCM/evaluators/transport
                     ./LCM/evaluators/HMC
                     ./LCM/models
                     ./LCM/parallel_evaluators
                     ./LCM/parallel_models
                     ./LCM/solvers
                     ./LCM/utils
                     ./LCM/utils/topology
                     ./LandIce/problems
                     ./LandIce/evaluators
                     ./TDM/problems
                     ./TDM/evaluators
                     ./Tsunami/problems
                     ./Tsunami/evaluators
                     ./Aeras/problems
                     ./Aeras/evaluators
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
  include_directories (${PlatoEngine_INSTALL_DIR}/include)
  set(Cogent_LIB ${PlatoEngine_INSTALL_DIR}/lib/libPlatoGeometryCogent.a)
ELSE()
  set(Cogent_LIB )
ENDIF()

IF ( ENABLE_MPMD )
  include_directories (${PlatoEngine_INSTALL_DIR}/include)
  set(PlatoEngine_LIB
      ${PlatoEngine_INSTALL_DIR}/lib/libPlatoDataLayer.a
      ${PlatoEngine_INSTALL_DIR}/lib/libPlatoInterface.a
      ${PlatoEngine_INSTALL_DIR}/lib/libPlatoAnalyze.a
      ${PlatoEngine_INSTALL_DIR}/lib/libPlatoParser.a
      ${PlatoEngine_INSTALL_DIR}/lib/libPlatoExceptions.a)
  ADD_DEFINITIONS(-DCODE_MP)
ELSE()
  set(PlatoEngine_LIB )
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

add_executable(Albany Main_Solve.cpp)

SET(ALBANY_EXECUTABLES Albany)

IF (ALBANY_MPMD)
  add_executable(AlbanyMPMD Main_Solve_MPMD.cpp)
  SET (ALBANY_EXECUTABLES ${ALBANY_EXECUTABLES} AlbanyMPMD )
ENDIF()

IF (ALBANY_DAKOTA)
  add_executable(AlbanyDakota Main_Dakota.cpp Albany_Dakota.cpp Albany_Dakota.hpp)
  SET(ALBANY_EXECUTABLES ${ALBANY_EXECUTABLES} AlbanyDakota)
ENDIF ()

add_executable(AlbanyAnalysis Main_Analysis.cpp)
SET(ALBANY_EXECUTABLES ${ALBANY_EXECUTABLES} AlbanyAnalysis)

IF (ALBANY_MESHDB_TOOLS)
  add_executable(exopumiconvert disc/tools/exopumiconvert.cpp)
  SET(ALBANY_EXECUTABLES ${ALBANY_EXECUTABLES} exopumiconvert)
ENDIF()

ENDIF (NOT ALBANY_LIBRARIES_ONLY)
# End declaration of executables

# Note: Albany_EXTRA_LIBRARIES is to allow users to specify
# libraries that one cannot get out of Trilinos directly.
set(ALL_LIBRARIES
  ${ALBANY_CI_LIB}
  ${PlatoEngine_LIB}
  ${PlatoEngine_LIB}
  ${Cogent_LIB}
  ${ALB_TRILINOS_LIBS}
  ${LAME_LIB}
  ${MATH_TOOLKIT_LIB}
  ${NLOPT_LIB}
  ${LAMENT_LIB}
  ${Trilinos_EXTRA_LD_FLAGS}
  ${Albany_EXTRA_LIBRARIES}
  ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}
  )

# Add Albany internal libraries/physics sets, as enabled.

add_subdirectory(adapt)
add_subdirectory(disc)

IF(ALBANY_STK)
SET(ALBANY_LIBRARIES albanyLib albanyAdapt albanySTK)
ELSE()
SET(ALBANY_LIBRARIES albanyLib albanyAdapt)
ENDIF()

IF(ALBANY_DEMO_PDES)
  add_library(albanyPDEs ${Albany_LIBRARY_TYPE} ${PDE_SRCS} ${PDE_HDRS})
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} albanyPDEs)
ENDIF()

IF (ALBANY_LANDICE)
  add_subdirectory(LandIce)
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} LandIce)

  IF (ENABLE_MPAS_INTERFACE)
    SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} mpasInterface)
  ENDIF()

  IF (ENABLE_CISM_INTERFACE)
    SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} cismInterface)
  ENDIF()
ENDIF()

IF (ALBANY_AERAS)
  add_subdirectory(Aeras)
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} Aeras)
ENDIF()

IF (ALBANY_TDM)
  add_subdirectory(TDM)
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} TDM)
ENDIF()

IF (ALBANY_TSUNAMI)
  add_subdirectory(Tsunami)
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} Tsunami)
ENDIF()

# RPI APF mesh interface
IF(ALBANY_APF)
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} albanyPUMI)
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

IF (ALBANY_LCM)
  add_subdirectory(LCM)
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} LCM)
ENDIF()

# Can remove this once rebalance is in Trilnos again
IF(ALBANY_STK_REBALANCE)
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} albanySTKRebalance)
endif()

# Repeat libraries twice to catch circular dependencies
SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} ${ALBANY_LIBRARIES} ${ALBANY_LIBRARIES})

MESSAGE("\n-- Albany Executables to link: ${ALBANY_EXECUTABLES}")

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
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Albany_config.h
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
