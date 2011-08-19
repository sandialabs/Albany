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
	ENAT_SGNOXSolver.cpp 
	Albany_SolverFactory.cpp
	Albany_Utils.cpp
	PHAL_Dimension.cpp 
	PHAL_AlbanyTraits.cpp 
	Albany_Application.cpp 
	Albany_RythmosObserver.cpp 
	Albany_NOXObserver.cpp 
	Albany_StateManager.cpp 
	Albany_ModelEvaluator.cpp
	Albany_SaveEigenData.cpp
	Albany_TimeManager.cpp
  )

SET(HEADERS
	ENAT_SGNOXSolver.hpp 
	Albany_SolverFactory.hpp
	Albany_Utils.hpp
	Albany_DataTypes.hpp 
	PHAL_Dimension.hpp 
	PHAL_FactoryTraits.hpp 
	PHAL_AlbanyTraits.hpp 
	PHAL_Workset.hpp 
	Albany_Application.hpp 
	Albany_RythmosObserver.hpp 
	Albany_NOXObserver.hpp 
	Albany_StateManager.hpp 
        Albany_StateInfoStruct.hpp
        Albany_EigendataInfoStruct.hpp
	Albany_ModelEvaluator.hpp
	Albany_SaveEigenData.hpp
        Albany_DummyParameterAccessor.hpp
        Albany_TimeManager.hpp
  )

#problems
SET(SOURCES ${SOURCES}
	problems/Albany_AbstractProblem.cpp 
	problems/Albany_ProblemFactory.cpp
	problems/Albany_HeatProblem.cpp 
	problems/Albany_NavierStokes.cpp 
	problems/Albany_ODEProblem.cpp 
	problems/Albany_Helmholtz2DProblem.cpp 
	problems/Albany_ThermoElectrostaticsProblem.cpp 
	problems/Albany_AnalyticFunction.cpp 
	problems/Albany_InitialCondition.cpp 
  )
SET(HEADERS ${HEADERS}
	problems/Albany_ProblemFactory.hpp 
	problems/Albany_AbstractProblem.hpp 
	problems/Albany_HeatProblem.hpp 
	problems/Albany_NavierStokes.hpp 
	problems/Albany_ODEProblem.hpp 
	problems/Albany_Helmholtz2DProblem.hpp 
	problems/Albany_ThermoElectrostaticsProblem.hpp 
	problems/Albany_AnalyticFunction.hpp 
	problems/Albany_InitialCondition.hpp 
  )

#responses
SET(SOURCES ${SOURCES}
	responses/Albany_SolutionAverageResponseFunction.cpp 
	responses/Albany_SolutionMaxValueResponseFunction.cpp
	responses/Albany_SolutionTwoNormResponseFunction.cpp
        responses/Albany_SolutionFileL2ResponseFunction.cpp
	responses/Albany_EvaluatedResponseFunction.cpp
  )
SET(HEADERS ${HEADERS}
	responses/Albany_AbstractResponseFunction.hpp 
	responses/Albany_SolutionAverageResponseFunction.hpp 
	responses/Albany_SolutionMaxValueResponseFunction.hpp
	responses/Albany_SolutionTwoNormResponseFunction.hpp
        responses/Albany_SolutionFileL2ResponseFunction.hpp
	responses/Albany_EvaluatedResponseFunction.hpp
  )

#stk
SET(SOURCES ${SOURCES}
        stk/Albany_STKDiscretization.cpp
	stk/Albany_DiscretizationFactory.cpp
        stk/Albany_GenericSTKMeshStruct.cpp
        stk/Albany_Point0DSTKMeshStruct.cpp
        stk/Albany_Line1DSTKMeshStruct.cpp
        stk/Albany_Rect2DSTKMeshStruct.cpp
        stk/Albany_Cube3DSTKMeshStruct.cpp
        stk/Albany_FromCubitSTKMeshStruct.cpp
        stk/Albany_IossSTKMeshStruct.cpp
  )
SET(HEADERS ${HEADERS}
	stk/Albany_AbstractDiscretization.hpp 
        stk/Albany_STKDiscretization.hpp
	stk/Albany_DiscretizationFactory.hpp
        stk/Albany_AbstractSTKMeshStruct.hpp
        stk/Albany_GenericSTKMeshStruct.hpp
        stk/Albany_Point0DSTKMeshStruct.hpp
        stk/Albany_Line1DSTKMeshStruct.hpp
        stk/Albany_Rect2DSTKMeshStruct.hpp
        stk/Albany_Cube3DSTKMeshStruct.hpp
        stk/Albany_FromCubitSTKMeshStruct.hpp
        stk/Albany_IossSTKMeshStruct.hpp
  )

#evaluators
SET(SOURCES ${SOURCES}
	evaluators/PHAL_GatherSolution.cpp 
	evaluators/PHAL_GatherEigenvectors.cpp
	evaluators/PHAL_ScatterResidual.cpp 
	evaluators/PHAL_Dirichlet.cpp 
	evaluators/PHAL_Source.cpp 
	evaluators/PHAL_Constant.cpp 
	evaluators/PHAL_ODEResid.cpp 
	evaluators/PHAL_ThermalConductivity.cpp 
	evaluators/PHAL_Absorption.cpp 
        evaluators/PHAL_HeatEqResid.cpp
        evaluators/PHAL_ComputeBasisFunctions.cpp
        evaluators/PHAL_DOFInterpolation.cpp
        evaluators/PHAL_DOFGradInterpolation.cpp
        evaluators/PHAL_DOFVecInterpolation.cpp
        evaluators/PHAL_DOFVecGradInterpolation.cpp
        evaluators/PHAL_MapToPhysicalFrame.cpp
        evaluators/PHAL_HelmholtzResid.cpp
        evaluators/PHAL_GatherCoordinateVector.cpp
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
	evaluators/PHAL_ResponseBase.cpp
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
        evaluators/PHAL_ResponseBase.hpp
	evaluators/PHAL_ResponseBase_Def.hpp
  )

# LCM
IF(ALBANY_LCM)
  SET(LCM_DIR ${Albany_SOURCE_DIR}/src/LCM)

  SET(HEADERS ${HEADERS}
    ${LCM_DIR}/LCM_FactoryTraits.hpp
  )
  
  # LCM problems
  SET(SOURCES ${SOURCES}
    ${LCM_DIR}/problems/ElasticityProblem.cpp 
    ${LCM_DIR}/problems/NonlinearElasticityProblem.cpp 
    ${LCM_DIR}/problems/ThermoElasticityProblem.cpp 
    ${LCM_DIR}/problems/GradientDamageProblem.cpp 
  )

  SET(HEADERS ${HEADERS}
    ${LCM_DIR}/problems/ElasticityProblem.hpp
    ${LCM_DIR}/problems/NonlinearElasticityProblem.hpp 
    ${LCM_DIR}/problems/ThermoElasticityProblem.hpp 
    ${LCM_DIR}/problems/GradientDamageProblem.hpp 
  )

  # LCM evaluators
  SET(SOURCES ${SOURCES}
    ${LCM_DIR}/evaluators/Stress.cpp 
    ${LCM_DIR}/evaluators/Strain.cpp 
    ${LCM_DIR}/evaluators/ElasticModulus.cpp 
    ${LCM_DIR}/evaluators/PoissonsRatio.cpp 
    ${LCM_DIR}/evaluators/DefGrad.cpp 
    ${LCM_DIR}/evaluators/RCG.cpp 
    ${LCM_DIR}/evaluators/LCG.cpp 
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
    ${LCM_DIR}/evaluators/DamageSource.cpp
    ${LCM_DIR}/evaluators/BulkModulus.cpp
    ${LCM_DIR}/evaluators/ShearModulus.cpp
    ${LCM_DIR}/evaluators/KfieldBC.cpp
    ${LCM_DIR}/evaluators/DislocationDensity.cpp
  )

  SET(HEADERS ${HEADERS}
    ${LCM_DIR}/evaluators/Stress.hpp 
    ${LCM_DIR}/evaluators/Stress_Def.hpp
    ${LCM_DIR}/evaluators/Strain.hpp 
    ${LCM_DIR}/evaluators/Strain_Def.hpp 
    ${LCM_DIR}/evaluators/ElasticModulus.hpp 
    ${LCM_DIR}/evaluators/ElasticModulus_Def.hpp 
    ${LCM_DIR}/evaluators/PoissonsRatio.hpp 
    ${LCM_DIR}/evaluators/PoissonsRatio_Def.hpp
    ${LCM_DIR}/evaluators/DefGrad.hpp 
    ${LCM_DIR}/evaluators/DefGrad_Def.hpp 
    ${LCM_DIR}/evaluators/RCG.hpp 
    ${LCM_DIR}/evaluators/RCG_Def.hpp 
    ${LCM_DIR}/evaluators/LCG.hpp 
    ${LCM_DIR}/evaluators/LCG_Def.hpp
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
    ${LCM_DIR}/evaluators/DamageSource.hpp 
    ${LCM_DIR}/evaluators/DamageSource_Def.hpp
    ${LCM_DIR}/evaluators/BulkModulus.hpp 
    ${LCM_DIR}/evaluators/BulkModulus_Def.hpp
    ${LCM_DIR}/evaluators/ShearModulus.hpp 
    ${LCM_DIR}/evaluators/ShearModulus_Def.hpp
    ${LCM_DIR}/evaluators/KfieldBC.hpp 
    ${LCM_DIR}/evaluators/KfieldBC_Def.hpp
    ${LCM_DIR}/evaluators/DislocationDensity.hpp
    ${LCM_DIR}/evaluators/DislocationDensity_Def.hpp
  )

  #LCM utils
  SET(SOURCES ${SOURCES}
    ${LCM_DIR}/utils/Geometry.cc
    ${LCM_DIR}/utils/Partition.cc
    ${LCM_DIR}/utils/Tensor.cc
    ${LCM_DIR}/utils/Topology.cc
  )

  SET(HEADERS ${HEADERS}
    ${LCM_DIR}/utils/Geometry.h
    ${LCM_DIR}/utils/Partition.h
    ${LCM_DIR}/utils/Tensor.h
    ${LCM_DIR}/utils/Topology.h 
  )

IF(ALBANY_LAME)
  SET(SOURCES ${SOURCES}
    ${LCM_DIR}/utils/LameUtils.cpp
    ${LCM_DIR}/evaluators/LameStress.cpp 
    ${LCM_DIR}/problems/LameProblem.cpp 
  )
  SET(HEADERS ${HEADERS}
    ${LCM_DIR}/utils/LameUtils.hpp
    ${LCM_DIR}/evaluators/LameStress.hpp 
    ${LCM_DIR}/evaluators/LameStress_Def.hpp 
    ${LCM_DIR}/problems/LameProblem.hpp
  )
ENDIF()

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
                     ./ ./evaluators ./problems ./responses ./stk ./LCM/utils
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
add_executable(AlbanyAdjoint Main_Adjoint.cpp)
add_executable(AlbanyCoupled Main_Coupled.cpp)
add_executable(AlbanySGAdjoint Main_SGAdjoint.cpp)
add_executable(AlbanySG Main_SGSolve.cpp)
IF (ALBANY_DAKOTA)
  add_executable(AlbanyDakota Main_Dakota.cpp Albany_Dakota.cpp Albany_Dakota.hpp)
ENDIF ()

IF (ALBANY_QCAD)
  add_executable(AlbanyQCAD Main_QCAD_Solve.cpp)
ENDIF ()

add_executable(AlbanyAnalysis Main_Analysis.cpp)

# Tests
add_executable(PartitionTest LCM/test/utils/PartitionTest.cc)
add_executable(UtilsTest LCM/test/utils/UtilsTest.cc)
IF (ALBANY_LCM)
add_executable(NodeUpdate LCM/test/utils/NodeUpdate.cc)
ENDIF ()

target_link_libraries(Albany          albanyLib ${ALB_TRILINOS_LIBS} ${ALB_CUBIT_LIBS} ${LAME_LIB} ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}) 
target_link_libraries(AlbanySG        albanyLib ${ALB_TRILINOS_LIBS} ${ALB_CUBIT_LIBS} ${LAME_LIB} ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}) 
target_link_libraries(AlbanyAdjoint   albanyLib ${ALB_TRILINOS_LIBS} ${ALB_CUBIT_LIBS} ${LAME_LIB} ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}) 
target_link_libraries(AlbanyCoupled   albanyLib ${ALB_TRILINOS_LIBS} ${ALB_CUBIT_LIBS} ${LAME_LIB} ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}) 
target_link_libraries(AlbanySGAdjoint   albanyLib ${ALB_TRILINOS_LIBS} ${ALB_CUBIT_LIBS} ${LAME_LIB} ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}) 
IF (ALBANY_DAKOTA)
  target_link_libraries(AlbanyDakota    albanyLib ${ALB_TRILINOS_LIBS}
                        ${ALB_CUBIT_LIBS} ${LAME_LIB} ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}) 
ENDIF ()

IF (ALBANY_QCAD)
target_link_libraries(AlbanyQCAD      albanyLib ${ALB_TRILINOS_LIBS} ${ALB_CUBIT_LIBS} ${LAME_LIB} ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}) 
ENDIF ()

target_link_libraries(AlbanyAnalysis  albanyLib ${ALB_TRILINOS_LIBS} ${ALB_CUBIT_LIBS} ${LAME_LIB} 
                      ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}) 

#Linking tests
target_link_libraries(PartitionTest albanyLib ${ALB_TRILINOS_LIBS} ${ALB_CUBIT_LIBS} ${LAME_LIB} ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}) 
target_link_libraries(UtilsTest albanyLib ${ALB_TRILINOS_LIBS} ${ALB_CUBIT_LIBS} ${LAME_LIB} ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
IF (ALBANY_LCM)
target_link_libraries(NodeUpdate albanyLib ${ALB_TRILINOS_LIBS} ${ALB_CUBIT_LIBS} ${LAME_LIB} ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}) 
ENDIF ()
