##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

SET(SOURCES
  Albany_SolverFactory.cpp
  Albany_Utils.cpp
  PHAL_Dimension.cpp
  PHAL_AlbanyTraits.cpp
  Albany_Application.cpp
  Albany_ObserverFactory.cpp
  Albany_RythmosObserver.cpp
  Albany_NOXObserver.cpp
  Albany_PiroObserver.cpp
  Albany_StateManager.cpp
  Albany_ModelFactory.cpp
  Albany_ModelEvaluator.cpp
  Albany_SaveEigenData.cpp
  Albany_ObserverImpl.cpp
  )

SET(HEADERS
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
  Albany_PiroObserver.hpp
  Albany_StateManager.hpp
  Albany_StateInfoStruct.hpp
  Albany_EigendataInfoStruct.hpp
  Albany_ModelFactory.hpp
  Albany_ModelEvaluator.hpp
  Albany_SaveEigenData.hpp
  Albany_ObserverImpl.hpp
  Albany_DummyParameterAccessor.hpp
  Albany_DistributedParameterLibrary.hpp
  Albany_DistributedParameterDerivativeOp.hpp
  Albany_DistributedParameterLibrary_Epetra.hpp
  )

#problems
SET(SOURCES ${SOURCES}
  problems/Albany_AbstractProblem.cpp
  problems/Albany_ProblemUtils.cpp
  problems/Albany_Layouts.cpp
  problems/Albany_EvaluatorUtils.cpp
  problems/Albany_ResponseUtilities.cpp
  problems/Albany_BCUtils.cpp
  problems/Albany_ProblemFactory.cpp
  problems/QCAD_MaterialDatabase.cpp
  problems/Albany_HeatProblem.cpp
  )
SET(PDE_SRCS
  problems/Albany_NavierStokes.cpp
  problems/Albany_GPAMProblem.cpp
  problems/Albany_LinComprNSProblem.cpp
  problems/Albany_ComprNSProblem.cpp
  problems/Albany_ODEProblem.cpp
  problems/Albany_Helmholtz2DProblem.cpp
  problems/Albany_CahnHillProblem.cpp
  problems/Albany_PNPProblem.cpp
  )
SET(HEADERS ${HEADERS}
  problems/Albany_ProblemFactory.hpp
  problems/Albany_AbstractProblem.hpp
  problems/Albany_ProblemUtils.hpp
  problems/Albany_Layouts.hpp
  problems/Albany_EvaluatorUtils.hpp
  problems/Albany_EvaluatorUtils_Def.hpp
  problems/Albany_ResponseUtilities.hpp
  problems/Albany_ResponseUtilities_Def.hpp
  problems/Albany_BCUtils.hpp
  problems/Albany_BCUtils_Def.hpp
  problems/QCAD_MaterialDatabase.hpp
  problems/Albany_HeatProblem.hpp
  )
SET(PDE_HDRS
  problems/Albany_NavierStokes.hpp
  problems/Albany_GPAMProblem.hpp
  problems/Albany_LinComprNSProblem.hpp
  problems/Albany_ComprNSProblem.hpp
  problems/Albany_ODEProblem.hpp
  problems/Albany_Helmholtz2DProblem.hpp
  problems/Albany_CahnHillProblem.hpp
  problems/Albany_PNPProblem.hpp
  )

#responses
SET(SOURCES ${SOURCES}
  responses/Epetra_GatherAllV.cpp
  responses/Epetra_ImportWithAlternateMap.cpp
  responses/Albany_SolutionCullingStrategy.cpp
  responses/Albany_ScalarResponseFunction.cpp
  responses/Albany_AggregateScalarResponseFunction.cpp
  responses/Albany_SamplingBasedScalarResponseFunction.cpp
  responses/Albany_SolutionAverageResponseFunction.cpp
  responses/Albany_SolutionMaxValueResponseFunction.cpp
  responses/Albany_SolutionTwoNormResponseFunction.cpp
  responses/Albany_SolutionValuesResponseFunction.cpp
  responses/Albany_SolutionFileResponseFunction.cpp
  responses/Albany_FieldManagerScalarResponseFunction.cpp
  responses/Albany_DistributedResponseFunction.cpp
  responses/Albany_SolutionResponseFunction.cpp
  responses/Albany_KLResponseFunction.cpp
  responses/Albany_ResponseFactory.cpp
  )
SET(HEADERS ${HEADERS}
  responses/Epetra_GatherAllV.hpp
  responses/Epetra_ImportWithAlternateMap.hpp
  responses/Albany_SolutionCullingStrategy.hpp
  responses/Albany_AbstractResponseFunction.hpp
  responses/Albany_ScalarResponseFunction.hpp
  responses/Albany_AggregateScalarResponseFunction.hpp
  responses/Albany_SamplingBasedScalarResponseFunction.hpp
  responses/Albany_SolutionAverageResponseFunction.hpp
  responses/Albany_SolutionMaxValueResponseFunction.hpp
  responses/Albany_SolutionTwoNormResponseFunction.hpp
  responses/Albany_SolutionValuesResponseFunction.hpp
  responses/Albany_SolutionFileResponseFunction.hpp
  responses/Albany_FieldManagerScalarResponseFunction.hpp
  responses/Albany_DistributedResponseFunction.hpp
  responses/Albany_SolutionResponseFunction.hpp
  responses/Albany_KLResponseFunction.hpp
  responses/Albany_ResponseFactory.hpp
  )

#discretization
SET(SOURCES ${SOURCES}
  disc/Albany_DiscretizationFactory.cpp
  disc/Albany_AbstractMeshStruct.cpp
  disc/Adapt_NodalDataBlock.cpp
  )
SET(HEADERS ${HEADERS}
  disc/Albany_AbstractDiscretization.hpp
  disc/Albany_AbstractMeshStruct.hpp
  disc/Albany_AbstractFieldContainer.hpp
  disc/Albany_AbstractNodeFieldContainer.hpp
  disc/Albany_DiscretizationFactory.hpp
  disc/Adapt_NodalDataBlock.hpp
  )

#evaluators
SET(SOURCES ${SOURCES}
  evaluators/PHAL_GatherSolution.cpp
  evaluators/PHAL_GatherEigenvectors.cpp
  evaluators/PHAL_GatherCoordinateVector.cpp
  evaluators/PHAL_GatherAuxData.cpp
  evaluators/PHAL_ScatterResidual.cpp
  evaluators/PHAL_Dirichlet.cpp
  evaluators/PHAL_DirichletCoordinateFunction.cpp
  evaluators/PHAL_IdentityCoordinateFunctionTraits.cpp
  evaluators/PHAL_Neumann.cpp
  evaluators/PHAL_ComputeBasisFunctions.cpp
  evaluators/PHAL_DOFInterpolation.cpp
  evaluators/PHAL_DOFGradInterpolation.cpp
  evaluators/PHAL_DOFVecInterpolation.cpp
  evaluators/PHAL_DOFVecGradInterpolation.cpp
  evaluators/PHAL_MapToPhysicalFrame.cpp
  evaluators/PHAL_SaveStateField.cpp
  evaluators/PHAL_LoadStateField.cpp
  evaluators/PHAL_SharedParameter.cpp
  evaluators/PHAL_ResponseFieldIntegral.cpp
  evaluators/PHAL_ScatterScalarResponse.cpp
  evaluators/PHAL_SeparableScatterScalarResponse.cpp
  evaluators/QCAD_ResponseFieldIntegral.cpp
  evaluators/QCAD_ResponseFieldValue.cpp
  evaluators/QCAD_ResponseFieldAverage.cpp
  evaluators/QCAD_ResponseSaveField.cpp
  evaluators/QCAD_ResponseCenterOfMass.cpp
  evaluators/QCAD_MeshRegion.cpp
  evaluators/QCAD_EvaluatorTools.cpp
  evaluators/PHAL_HeatEqResid.cpp
  evaluators/PHAL_Source.cpp
  evaluators/PHAL_ThermalConductivity.cpp
  evaluators/PHAL_NSMaterialProperty.cpp
  evaluators/PHAL_Absorption.cpp
  evaluators/PHAL_GatherSHeight.cpp
  evaluators/PHAL_GatherTemperature.cpp
  evaluators/PHAL_GatherBasalFriction.cpp
  evaluators/PHAL_GatherThickness.cpp
  evaluators/PHAL_GatherFlowFactor.cpp
  evaluators/Adapt_ElementSizeField.cpp
  evaluators/PHAL_SaveNodalField.cpp
  evaluators/PHAL_GatherSurfaceVelocity.cpp
  evaluators/PHAL_GatherVelocityRMS.cpp
  evaluators/PHAL_GatherScalarNodalParameter.cpp
  )
SET(PDE_SRCS ${PDE_SRCS}
  evaluators/PHAL_Constant.cpp
  evaluators/PHAL_ODEResid.cpp
  evaluators/PHAL_GPAMResid.cpp
  evaluators/PHAL_LinComprNSResid.cpp
  evaluators/PHAL_LinComprNSBodyForce.cpp
  evaluators/PHAL_ComprNSResid.cpp
  evaluators/PHAL_ComprNSBodyForce.cpp
  evaluators/PHAL_ComprNSViscosity.cpp
  evaluators/PHAL_HelmholtzResid.cpp
  evaluators/PHAL_CahnHillRhoResid.cpp
  evaluators/PHAL_CahnHillWResid.cpp
  evaluators/PHAL_CahnHillChemTerm.cpp
  evaluators/PHAL_LangevinNoiseTerm.cpp
  evaluators/PHAL_NSContinuityResid.cpp
  evaluators/PHAL_NSMomentumResid.cpp
  evaluators/PHAL_NSThermalEqResid.cpp
  evaluators/PHAL_NSNeutronEqResid.cpp
  evaluators/PHAL_NSRm.cpp
  evaluators/PHAL_NSContravarientMetricTensor.cpp
  evaluators/PHAL_NSTauM.cpp
  evaluators/PHAL_NSTauT.cpp
  evaluators/PHAL_NSBodyForce.cpp
  evaluators/PHAL_NSPermeabilityTerm.cpp
  evaluators/PHAL_NSForchheimerTerm.cpp
  evaluators/PNP_PotentialResid.cpp
  evaluators/PNP_ConcentrationResid.cpp
  )

SET(HEADERS ${HEADERS}
  evaluators/PHAL_GatherSolution.hpp
  evaluators/PHAL_GatherSolution_Def.hpp
  evaluators/PHAL_GatherEigenvectors.hpp
  evaluators/PHAL_GatherEigenvectors_Def.hpp
  evaluators/PHAL_GatherCoordinateVector.hpp
  evaluators/PHAL_GatherCoordinateVector_Def.hpp
  evaluators/PHAL_GatherAuxData.hpp
  evaluators/PHAL_GatherAuxData_Def.hpp
  evaluators/PHAL_ScatterResidual.hpp
  evaluators/PHAL_ScatterResidual_Def.hpp
  evaluators/PHAL_Dirichlet.hpp
  evaluators/PHAL_Dirichlet_Def.hpp
  evaluators/PHAL_DirichletCoordinateFunction.hpp
  evaluators/PHAL_DirichletCoordinateFunction_Def.hpp
  evaluators/PHAL_IdentityCoordinateFunctionTraits.hpp
  evaluators/PHAL_IdentityCoordinateFunctionTraits_Def.hpp
  evaluators/PHAL_Neumann.hpp
  evaluators/PHAL_Neumann_Def.hpp
  evaluators/PHAL_Constant.hpp
  evaluators/PHAL_Constant_Def.hpp
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
  evaluators/PHAL_SaveStateField.hpp
  evaluators/PHAL_SaveStateField_Def.hpp
  evaluators/PHAL_LoadStateField.hpp
  evaluators/PHAL_LoadStateField_Def.hpp
  evaluators/PHAL_SharedParameter.hpp
  evaluators/PHAL_SharedParameter_Def.hpp
  evaluators/PHAL_ResponseFieldIntegral.hpp
  evaluators/PHAL_ResponseFieldIntegral_Def.hpp
  evaluators/PHAL_ScatterScalarResponse.hpp
  evaluators/PHAL_ScatterScalarResponse_Def.hpp
  evaluators/PHAL_SeparableScatterScalarResponse.hpp
  evaluators/PHAL_SeparableScatterScalarResponse_Def.hpp
  evaluators/QCAD_ResponseFieldIntegral.hpp
  evaluators/QCAD_ResponseFieldIntegral_Def.hpp
  evaluators/QCAD_ResponseFieldValue.hpp
  evaluators/QCAD_ResponseFieldValue_Def.hpp
  evaluators/QCAD_ResponseFieldAverage.hpp
  evaluators/QCAD_ResponseFieldAverage_Def.hpp
  evaluators/QCAD_ResponseSaveField.hpp
  evaluators/QCAD_ResponseSaveField_Def.hpp
  evaluators/QCAD_ResponseCenterOfMass.hpp
  evaluators/QCAD_ResponseCenterOfMass_Def.hpp
  evaluators/QCAD_MeshRegion.hpp
  evaluators/QCAD_MeshRegion_Def.hpp
  evaluators/QCAD_EvaluatorTools.hpp
  evaluators/QCAD_EvaluatorTools_Def.hpp
  evaluators/PHAL_HeatEqResid.hpp
  evaluators/PHAL_HeatEqResid_Def.hpp
  evaluators/PHAL_Source.hpp
  evaluators/PHAL_Source_Def.hpp
  evaluators/PHAL_ThermalConductivity.hpp
  evaluators/PHAL_ThermalConductivity_Def.hpp
  evaluators/PHAL_NSMaterialProperty.hpp
  evaluators/PHAL_NSMaterialProperty_Def.hpp
  evaluators/PHAL_Absorption.hpp
  evaluators/PHAL_Absorption_Def.hpp
  evaluators/PHAL_GatherSHeight.hpp
  evaluators/PHAL_GatherSHeight_Def.hpp
  evaluators/PHAL_GatherTemperature.hpp
  evaluators/PHAL_GatherTemperature_Def.hpp
  evaluators/PHAL_GatherBasalFriction.hpp
  evaluators/PHAL_GatherBasalFriction_Def.hpp
  evaluators/PHAL_GatherThickness.hpp
  evaluators/PHAL_GatherThickness_Def.hpp
  evaluators/PHAL_GatherFlowFactor.hpp
  evaluators/PHAL_GatherFlowFactor_Def.hpp
  evaluators/Adapt_ElementSizeField.hpp
  evaluators/Adapt_ElementSizeField_Def.hpp
  evaluators/PHAL_SaveNodalField.hpp
  evaluators/PHAL_SaveNodalField_Def.hpp
  evaluators/PHAL_GatherSurfaceVelocity.hpp
  evaluators/PHAL_GatherSurfaceVelocity_Def.hpp
  evaluators/PHAL_GatherVelocityRMS.hpp
  evaluators/PHAL_GatherVelocityRMS_Def.hpp
  evaluators/PHAL_GatherScalarNodalParameter.hpp
  evaluators/PHAL_GatherScalarNodalParameter_Def.hpp
  )

SET(PDE_HDRS ${PDE_HDRS}
  evaluators/PHAL_ODEResid.hpp
  evaluators/PHAL_ODEResid_Def.hpp
  evaluators/PHAL_GPAMResid.hpp
  evaluators/PHAL_GPAMResid_Def.hpp
  evaluators/PHAL_LinComprNSResid.hpp
  evaluators/PHAL_LinComprNSResid_Def.hpp
  evaluators/PHAL_ComprNSResid.hpp
  evaluators/PHAL_ComprNSResid_Def.hpp
  evaluators/PHAL_LinComprNSBodyForce.hpp
  evaluators/PHAL_LinComprNSBodyForce_Def.hpp
  evaluators/PHAL_ComprNSBodyForce.hpp
  evaluators/PHAL_ComprNSBodyForce_Def.hpp
  evaluators/PHAL_ComprNSViscosity.hpp
  evaluators/PHAL_ComprNSViscosity_Def.hpp
  evaluators/PHAL_ComputeBasisFunctions.hpp
  evaluators/PHAL_ComputeBasisFunctions_Def.hpp
  evaluators/PHAL_HelmholtzResid.hpp
  evaluators/PHAL_HelmholtzResid_Def.hpp
  evaluators/PHAL_CahnHillRhoResid.hpp
  evaluators/PHAL_CahnHillRhoResid_Def.hpp
  evaluators/PHAL_CahnHillWResid.hpp
  evaluators/PHAL_CahnHillWResid_Def.hpp
  evaluators/PHAL_CahnHillChemTerm.hpp
  evaluators/PHAL_CahnHillChemTerm_Def.hpp
  evaluators/PHAL_LangevinNoiseTerm.hpp
  evaluators/PHAL_LangevinNoiseTerm_Def.hpp
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
  evaluators/PHAL_NSBodyForce.hpp
  evaluators/PHAL_NSBodyForce_Def.hpp
  evaluators/PHAL_NSPermeabilityTerm.hpp
  evaluators/PHAL_NSPermeabilityTerm_Def.hpp
  evaluators/PHAL_NSForchheimerTerm.hpp
  evaluators/PHAL_NSForchheimerTerm_Def.hpp
  evaluators/PNP_PotentialResid.hpp
  evaluators/PNP_PotentialResid_Def.hpp
  evaluators/PNP_ConcentrationResid.hpp
  evaluators/PNP_ConcentrationResid_Def.hpp
  )

# FELIX
IF (ALBANY_FELIX)
  SET(SOURCES ${SOURCES}
    evaluators/FELIX_ResponseSurfaceVelocityMismatch.cpp
  )
  SET(HEADERS ${HEADERS}
    evaluators/FELIX_ResponseSurfaceVelocityMismatch.hpp
    evaluators/FELIX_ResponseSurfaceVelocityMismatch_Def.hpp
  )
ENDIF()

# ASCR
IF(ALBANY_ASCR)
  SET(SOURCES ${SOURCES}
    Albany_Networks.cpp
    )
  SET(HEADERS ${HEADERS}
    Albany_Networks.hpp
    )
ENDIF()

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

if (ALBANY_MOR)
  include_directories (./MOR)

  set (MOR_ALBANY_HEADERS
    MOR/Albany_MORFacade.hpp
    MOR/Albany_MORFacadeImpl.hpp
    MOR/Albany_StkEpetraMVSource.hpp
    MOR/Albany_DiscretizationDofListProvider.hpp
    MOR/Albany_StkNodalBasisSource.hpp
    MOR/Albany_MORDiscretizationUtils.hpp
    )

  set (MOR_ALBANY_SOURCES
    MOR/Albany_MORFacadeImpl.cpp
    MOR/Albany_StkEpetraMVSource.cpp
    MOR/Albany_DiscretizationDofListProvider.cpp
    MOR/Albany_StkNodalBasisSource.cpp
    MOR/Albany_MORDiscretizationUtils.cpp
    )

  set(SOURCES ${SOURCES} ${MOR_ALBANY_SOURCES})
  set(HEADERS ${HEADERS} ${MOR_ALBANY_HEADERS})
endif ()


add_library(albanyLib ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})


# Add Albany external libraries

set(ALB_TRILINOS_LIBS ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})

include_directories (${Trilinos_INCLUDE_DIRS}
                     ${Trilinos_TPL_INCLUDE_DIRS}
                     ${ALBANY_FMDB_INCLUDE_DIRS}
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
                     ./LCM/evaluators/surface-element
                     ./LCM/models
                     ./LCM/utils
                     ./LCM/utils/topology
                     ./FELIX/problems
                     ./FELIX/evaluators
                     ./Aeras/problems
                     ./Aeras/evaluators
                     ./Hydride/problems
                     ./Hydride/evaluators
                     ./QCAD
                     ./QCAD/problems
                     ./QCAD/evaluators
                     ./QCAD/responses
                    )

link_directories (${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS}
  ${FMDB_LIBRARY_DIR})

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

# PERIDIGM
IF(ALBANY_PERIDIGM)
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

# RPI SCOREC Adaptivity
IF(ALBANY_SCOREC)
##  set(SCOREC_LIB FMDB SCORECModel SCORECUtil ipcomman parmetis metis)
## parmetis and metis compiled into trilinos and contained in $Trilinos_TPL_LIBRARIES
#  IF(DEFINED ENABLE_PARASOLID AND DEFINED ENABLE_ACIS)
#    set(SCOREC_LIB GMIParasolid pskernel GMIACIS hoops1700 SpaACIS Xext X11 FMDB MeshAdapt meshModel SCORECModel SCORECUtil ipcomman FMDB)
#  ELSEIF(ENABLE_PARASOLID)
##    set(SCOREC_LIB GMIParasolid pskernel FMDB MeshAdapt meshModel SCORECModel SCORECUtil ipcomman)
##    set(SCOREC_LIB GMIParasolid pskernel FMDB MeshAdapt meshModel SCORECModel SCORECUtil ipcomman FMDB)
#  ELSEIF(ENABLE_ACIS)
#    set(SCOREC_LIB GMIACIS hoops1700 SpaACIS Xext X11 FMDB MeshAdapt meshModel SCORECModel SCORECUtil ipcomman)
#  ELSE()
##    set(SCOREC_LIB FMDB MeshAdapt GMIMeshModel SCORECUtil ipcomman)
#  ENDIF()
#  # postpend boost libraries to support FMDB serialization calls
##    set(SCOREC_LIB ${SCOREC_LIB} boost_wserialization boost_serialization boost_mpi)

  SET(ALBANY_FMDB_INCLUDE_DIRS ./disc/fmdb)
  SET(FMDB_INCLUDE_DIR "${TRILINOS_INCLUDE_DIRS}/SCOREC")
  include_directories (${ALBANY_FMDB_INCLUDE_DIRS} ${FMDB_INCLUDE_DIR})

ELSE()
  set(SCOREC_LIB )
  SET(ALBANY_FMDB_INCLUDE_DIRS )
ENDIF()

# Catalyst CoProcessing
if(ALBANY_CATALYST)
  set(ALBANY_CATALYST_INCLUDE_DIRS ./disc/catalyst)
  include_directories(${ALBANY_CATALYST_INCLUDE_DIRS})
endif()

# Now build executables from this library, tailored main()s, and Trilinos

add_executable(Albany Main_Solve.cpp)
SET(ALBANY_EXECUTABLES Albany )
if (ALBANY_SG_MP)
add_executable(AlbanySG Main_SGSolve.cpp)
SET(ALBANY_EXECUTABLES ${ALBANY_EXECUTABLES} AlbanySG )
endif()
if (ALBANY_ASCR)
  add_executable(AlbanyAdjoint Main_Adjoint.cpp)
  add_executable(AlbanySGAdjoint Main_SGAdjoint.cpp)
  add_executable(AlbanyCoupled Main_Coupled.cpp)
  add_executable(AlbanySGCoupled Main_SGCoupled.cpp)
SET(ALBANY_EXECUTABLES ${ALBANY_EXECUTABLES}
    AlbanyAdjoint
    AlbanySGAdjoint
    AlbanyCoupled
    AlbanySGCoupled
    )
endif()
IF (ALBANY_DAKOTA)
  add_executable(AlbanyDakota Main_Dakota.cpp Albany_Dakota.cpp Albany_Dakota.hpp)
  SET(ALBANY_EXECUTABLES ${ALBANY_EXECUTABLES} AlbanyDakota)
ENDIF ()

IF (ALBANY_MOR)
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

add_executable(AlbanyAnalysis Main_Analysis.cpp)
SET(ALBANY_EXECUTABLES ${ALBANY_EXECUTABLES} AlbanyAnalysis)

set(ALL_LIBRARIES
  ${ALBANY_CI_LIB}
  ${ALB_TRILINOS_LIBS}
  ${ALB_CUBIT_LIBS}
  ${LAME_LIB}
  ${LAMENT_LIB}
  ${Peridigm_LIBRARIES}
  ${SCOREC_LIB}
  ${Trilinos_EXTRA_LD_FLAGS}
  ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}
  )

# Add Albany internal libraries/physics sets, as enabled.

add_subdirectory(adapt)
add_subdirectory(disc)

SET(ALBANY_LIBRARIES albanyLib albanyAdapt albanySTK)

IF(ALBANY_DEMO_PDES)
  add_library(albanyPDEs ${Albany_LIBRARY_TYPE} ${PDE_SRCS} ${PDE_HDRS})
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} albanyPDEs)
ENDIF()

IF (ALBANY_MOR)
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
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} mpasInterface)
  IF (ENABLE_CISM_INTERFACE)
    SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} cismInterface)
  ENDIF()
ENDIF()

IF (ALBANY_AERAS)
  add_subdirectory(Aeras)
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} Aeras)
ENDIF()

# RPI SCOREC Adaptivity tools
IF(ALBANY_SCOREC)

  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} fmdb)

ENDIF()

IF(ALBANY_CATALYST)
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} albanyCatalyst)
endif()

IF (ALBANY_LCM)
  add_subdirectory(LCM)

  IF (ALBANY_HYDRIDE)
    #Hydride requires LCM
    add_subdirectory(Hydride)
    SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} hydride LCM)
  ELSE()
    SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} LCM)
  ENDIF()

ENDIF()

# Repeat libraries twice to catch circular dependencies
SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} ${ALBANY_LIBRARIES})

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

ENDIF()
