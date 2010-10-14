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
	PHAL_Dimension.cpp 
	PHAL_AlbanyTraits.cpp 
	Albany_Application.cpp 
	Albany_RythmosObserver.cpp 
	Albany_NOXObserver.cpp 
	Albany_StateManager.cpp 
	Albany_VTK.cpp 
	Albany_ModelEvaluator.cpp
  )

SET(HEADERS
	ENAT_SGNOXSolver.hpp 
	Albany_SolverFactory.hpp
	Albany_DataTypes.hpp 
	PHAL_Dimension.hpp 
	PHAL_FactoryTraits.hpp 
	PHAL_AlbanyTraits.hpp 
	PHAL_Workset.hpp 
	Albany_Application.hpp 
	Albany_RythmosObserver.hpp 
	Albany_NOXObserver.hpp 
	Albany_StateManager.hpp 
	Albany_VTK.hpp 
	Albany_ModelEvaluator.hpp
        CUTR_AbstractMeshMover.hpp
        Albany_DummyParameterAccessor.hpp
  )

#problems
SET(SOURCES ${SOURCES}
	problems/Albany_AbstractProblem.cpp 
	problems/Albany_ProblemFactory.cpp
	problems/Albany_HeatProblem.cpp 
	problems/Albany_Helmholtz2DProblem.cpp 
	problems/Albany_ElasticityProblem.cpp 
	problems/Albany_NonlinearElasticityProblem.cpp 
	problems/Albany_ThermoElasticityProblem.cpp 
	problems/Albany_AnalyticFunction.cpp 
	problems/Albany_InitialCondition.cpp 
  )
SET(HEADERS ${HEADERS}
	problems/Albany_ProblemFactory.hpp 
	problems/Albany_AbstractProblem.hpp 
	problems/Albany_HeatProblem.hpp 
	problems/Albany_Helmholtz2DProblem.hpp 
	problems/Albany_ElasticityProblem.hpp 
	problems/Albany_NonlinearElasticityProblem.hpp 
	problems/Albany_ThermoElasticityProblem.hpp 
	problems/Albany_AnalyticFunction.hpp 
	problems/Albany_InitialCondition.hpp 
  )

#responses
SET(SOURCES ${SOURCES}
	responses/Albany_BoundaryFlux1DResponseFunction.cpp 
	responses/Albany_SolutionAverageResponseFunction.cpp 
	responses/Albany_SolutionTwoNormResponseFunction.cpp
  )
SET(HEADERS ${HEADERS}
	responses/Albany_AbstractResponseFunction.hpp 
	responses/Albany_BoundaryFlux1DResponseFunction.hpp 
	responses/Albany_SolutionAverageResponseFunction.hpp 
	responses/Albany_SolutionTwoNormResponseFunction.hpp
  )

#stk
SET(SOURCES ${SOURCES}
        stk/Albany_STKDiscretization.cpp
	stk/Albany_DiscretizationFactory.cpp
        stk/Albany_Line1DSTKMeshStruct.cpp
        stk/Albany_Quad2DSTKMeshStruct.cpp
        stk/Albany_Cube3DSTKMeshStruct.cpp
        stk/Albany_FromCubitSTKMeshStruct.cpp
        stk/Albany_IossSTKMeshStruct.cpp
  )
SET(HEADERS ${HEADERS}
	stk/Albany_AbstractDiscretization.hpp 
        stk/Albany_STKDiscretization.hpp
	stk/Albany_DiscretizationFactory.hpp
        stk/Albany_AbstractSTKMeshStruct.hpp
        stk/Albany_Line1DSTKMeshStruct.hpp
        stk/Albany_Quad2DSTKMeshStruct.hpp
        stk/Albany_Cube3DSTKMeshStruct.hpp
        stk/Albany_FromCubitSTKMeshStruct.hpp
        stk/Albany_IossSTKMeshStruct.hpp
  )

#evaluators
SET(SOURCES ${SOURCES}
	evaluators/PHAL_GatherSolution.cpp 
	evaluators/PHAL_ScatterResidual.cpp 
	evaluators/PHAL_Dirichlet.cpp 
	evaluators/PHAL_Source.cpp 
	evaluators/PHAL_Constant.cpp 
	evaluators/PHAL_ThermalConductivity.cpp 
        evaluators/PHAL_HeatEqResid.cpp
        evaluators/PHAL_ElasticityResid.cpp
        evaluators/PHAL_ComputeBasisFunctions.cpp
        evaluators/PHAL_DOFInterpolation.cpp
        evaluators/PHAL_DOFGradInterpolation.cpp
        evaluators/PHAL_DOFVecInterpolation.cpp
        evaluators/PHAL_DOFVecGradInterpolation.cpp
        evaluators/PHAL_MapToPhysicalFrame.cpp
        evaluators/PHAL_HelmholtzResid.cpp
        evaluators/PHAL_GatherCoordinateVector.cpp
	evaluators/PHAL_Stress.cpp 
	evaluators/PHAL_Strain.cpp 
	evaluators/PHAL_ElasticModulus.cpp 
	evaluators/PHAL_PoissonsRatio.cpp
	evaluators/PHAL_DefGrad.cpp
	evaluators/PHAL_RCG.cpp
	evaluators/PHAL_LCG.cpp 
	evaluators/PHAL_DetDefGrad.cpp
	evaluators/PHAL_Neohookean.cpp
	evaluators/PHAL_J2Stress.cpp
	evaluators/PHAL_TLElasResid.cpp
  )
SET(HEADERS ${HEADERS}
	evaluators/PHAL_GatherSolution.hpp 
	evaluators/PHAL_GatherSolution_Def.hpp 
	evaluators/PHAL_ScatterResidual.hpp 
	evaluators/PHAL_ScatterResidual_Def.hpp 
	evaluators/PHAL_Dirichlet.hpp 
	evaluators/PHAL_Dirichlet_Def.hpp 
	evaluators/PHAL_Source.hpp 
	evaluators/PHAL_Source_Def.hpp 
	evaluators/PHAL_Constant.hpp 
	evaluators/PHAL_Constant_Def.hpp 
	evaluators/PHAL_ThermalConductivity.hpp 
	evaluators/PHAL_ThermalConductivity_Def.hpp
        evaluators/PHAL_HeatEqResid.hpp
        evaluators/PHAL_HeatEqResid_Def.hpp
        evaluators/PHAL_ElasticityResid.hpp
        evaluators/PHAL_ElasticityResid_Def.hpp
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
	evaluators/PHAL_Stress.hpp 
	evaluators/PHAL_Stress_Def.hpp 
	evaluators/PHAL_Strain.hpp 
	evaluators/PHAL_Strain_Def.hpp 
	evaluators/PHAL_ElasticModulus.hpp 
	evaluators/PHAL_ElasticModulus_Def.hpp 
	evaluators/PHAL_PoissonsRatio.hpp 
	evaluators/PHAL_PoissonsRatio_Def.hpp
	evaluators/PHAL_DefGrad.hpp 
	evaluators/PHAL_DefGrad_Def.hpp 
	evaluators/PHAL_RCG.hpp 
	evaluators/PHAL_RCG_Def.hpp 
	evaluators/PHAL_LCG.hpp 
	evaluators/PHAL_LCG_Def.hpp
	evaluators/PHAL_DetDefGrad.hpp 
	evaluators/PHAL_DetDefGrad_Def.hpp
	evaluators/PHAL_Neohookean.hpp
	evaluators/PHAL_Neohookean_Def.hpp
	evaluators/PHAL_J2Stress.hpp 
	evaluators/PHAL_J2Stress_Def.hpp
	evaluators/PHAL_TLElasResid.hpp 
	evaluators/PHAL_TLElasResid_Def.hpp 
  )

SET(SOURCES ${SOURCES}
       evaluators/QCAD_Permittivity.cpp
       evaluators/QCAD_PoissonResid.cpp
       evaluators/QCAD_PoissonSource.cpp
       problems/QCAD_PoissonProblem.cpp
  )

SET(HEADERS ${HEADERS}
       evaluators/QCAD_Permittivity.hpp
       evaluators/QCAD_Permittivity_Def.hpp
       evaluators/QCAD_PoissonResid.hpp
       evaluators/QCAD_PoissonResid_Def.hpp
       evaluators/QCAD_PoissonSource.hpp
       evaluators/QCAD_PoissonSource_Def.hpp
       problems/QCAD_PoissonProblem.hpp
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
                     ${Trilinos_TPL_INCLUDE_DIRS})

link_directories (${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})

if (ALBANY_ENABLE_CUBIT)
  INCLUDE(${MESHMOVER_DIR}/UseUtil.cmake)
  set (SOURCES ${SOURCES} ${CUTR_SRCS})
  set (HEADERS ${HEADERS} ${CUTR_HDRS})
  include_directories (${MESHMOVER_DIR} ${MESHMOVER_DIR}/../cubit)
  link_directories (${MESHMOVER_DIR}/../cubit/linux64)
  set(ALB_CUBIT_LIBS  cubiti19)
  ADD_DEFINITIONS(-DALBANY_CUTR)
endif (ALBANY_ENABLE_CUBIT)

add_library(albanyLib ${SOURCES} ${HEADERS})
# Now build executables from this library, tailored main()s, and Trilinos

add_executable(Albany Main_Solve.cpp)
add_executable(AlbanySG Main_SGSolve.cpp)
IF (ALBANY_DAKOTA)
  add_executable(AlbanyDakota Main_Dakota.cpp Albany_Dakota.cpp Albany_Dakota.hpp)
ENDIF ()
add_executable(AlbanyAnalysis Main_Analysis.cpp)

target_link_libraries(Albany          albanyLib ${ALB_TRILINOS_LIBS} ${ALB_CUBIT_LIBS}) 
target_link_libraries(AlbanySG        albanyLib ${ALB_TRILINOS_LIBS} ${ALB_CUBIT_LIBS}) 
IF (ALBANY_DAKOTA)
  target_link_libraries(AlbanyDakota    albanyLib ${ALB_TRILINOS_LIBS}
                        ${ALB_CUBIT_LIBS} ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}) 
ENDIF ()

target_link_libraries(AlbanyAnalysis  albanyLib ${ALB_TRILINOS_LIBS} ${ALB_CUBIT_LIBS}
                      ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}) 
