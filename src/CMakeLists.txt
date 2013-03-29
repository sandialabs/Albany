##*****************************************************************//

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
  Albany_StateManager.cpp
  Albany_ModelFactory.cpp
  Albany_ModelEvaluator.cpp
  Albany_SaveEigenData.cpp
  Albany_ExodusOutput.cpp
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
  problems/Albany_ProblemUtils.cpp
  problems/Albany_Layouts.cpp
  problems/Albany_EvaluatorUtils.cpp
  problems/Albany_ResponseUtilities.cpp
  problems/Albany_BCUtils.cpp
  problems/Albany_ProblemFactory.cpp
  problems/Albany_AnalyticFunction.cpp
  problems/Albany_InitialCondition.cpp
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
  problems/Albany_AnalyticFunction.hpp
  problems/Albany_InitialCondition.hpp
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
  )

#responses
SET(SOURCES ${SOURCES}
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
  )
SET(HEADERS ${HEADERS}
  disc/Albany_AbstractDiscretization.hpp
  disc/Albany_AbstractMeshStruct.hpp
  disc/Albany_DiscretizationFactory.hpp
  )

#evaluators
SET(SOURCES ${SOURCES}
  evaluators/PHAL_GatherSolution.cpp
  evaluators/PHAL_GatherEigenvectors.cpp
  evaluators/PHAL_GatherCoordinateVector.cpp
  evaluators/PHAL_ScatterResidual.cpp
  evaluators/PHAL_Dirichlet.cpp
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
  )

SET(HEADERS ${HEADERS}
  evaluators/PHAL_GatherSolution.hpp
  evaluators/PHAL_GatherSolution_Def.hpp
  evaluators/PHAL_GatherEigenvectors.hpp
  evaluators/PHAL_GatherEigenvectors_Def.hpp
  evaluators/PHAL_GatherCoordinateVector.hpp
  evaluators/PHAL_GatherCoordinateVector_Def.hpp
  evaluators/PHAL_ScatterResidual.hpp
  evaluators/PHAL_ScatterResidual_Def.hpp
  evaluators/PHAL_Dirichlet.hpp
  evaluators/PHAL_Dirichlet_Def.hpp
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


add_library(albanyLib ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})


# Add Albany external libraries

set(ALB_TRILINOS_LIBS ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})

include_directories (${Trilinos_INCLUDE_DIRS}
  ${Trilinos_TPL_INCLUDE_DIRS}
  ${ALBANY_FMDB_INCLUDE_DIRS}
  ./ ./evaluators ./problems
  ./responses ./disc/stk ./adapt ./disc
  ./LCM/ ./LCM/evaluators ./LCM/utils ./LCM/utils/tensor
  ./Hydride/problems ./Hydride/evaluators
  ./FELIX/problems ./FELIX/evaluators
  ./QCAD ./QCAD/problems ./QCAD/evaluators ./QCAD/responses
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
  #  set(SCOREC_LIB FMDB SCORECModel SCORECUtil ipcomman parmetis metis)
  # parmetis and metis compiled into trilinos and contained in $Trilinos_TPL_LIBRARIES
  set(SCOREC_LIB FMDB SCORECModel SCORECUtil ipcomman)
ELSE()
  set(SCOREC_LIB )
ENDIF()

# Now build executables from this library, tailored main()s, and Trilinos

add_executable(Albany Main_Solve.cpp)
if (ALBANY_SG_MP)
add_executable(AlbanySG Main_SGSolve.cpp)
endif()
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

set(ALL_LIBRARIES 
  ${ALBANY_CI_LIB} 
  ${ALB_TRILINOS_LIBS} 
  ${ALB_CUBIT_LIBS}
  ${LAME_LIB} 
  ${LAMENT_LIB} 
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
ENDIF()

# RPI SCOREC Adaptivity tools
IF(ALBANY_SCOREC)

  SET(ALBANY_FMDB_INCLUDE_DIRS ./disc/fmdb)
  SET(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} fmdb)
  include_directories (${ALBANY_FMDB_INCLUDE_DIRS} ${FMDB_INCLUDE_DIR})

ELSE()

  SET(ALBANY_FMDB_INCLUDE_DIRS )

ENDIF()

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

target_link_libraries(Albany          ${ALBANY_LIBRARIES}  ${ALL_LIBRARIES})
if (ALBANY_SG_MP)
target_link_libraries(AlbanySG        ${ALBANY_LIBRARIES}  ${ALL_LIBRARIES})
endif()
if (ALBANY_ASCR)
  target_link_libraries(AlbanyAdjoint   ${ALBANY_LIBRARIES}  ${ALL_LIBRARIES})
  target_link_libraries(AlbanySGAdjoint   ${ALBANY_LIBRARIES}  ${ALL_LIBRARIES})
  target_link_libraries(AlbanyCoupled   ${ALBANY_LIBRARIES}  ${ALL_LIBRARIES})
  target_link_libraries(AlbanySGCoupled   ${ALBANY_LIBRARIES}  ${ALL_LIBRARIES})
endif()
IF (ALBANY_DAKOTA)
  target_link_libraries(AlbanyDakota    ${ALBANY_LIBRARIES}  ${ALL_LIBRARIES})
ENDIF ()

target_link_libraries(AlbanyAnalysis  ${ALBANY_LIBRARIES}  ${ALL_LIBRARIES})

IF (INSTALL_ALBANY)
  configure_package_config_file(AlbanyConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/AlbanyConfig.cmake
    INSTALL_DESTINATION ${LIB_INSTALL_DIR}/Albany/cmake
    PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR)
  install(FILES ${HEADERS}
    DESTINATION ${INCLUDE_INSTALL_DIR})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/AlbanyConfig.cmake
    DESTINATION ${LIB_INSTALL_DIR}/Albany/cmake )
  install(TARGETS albanyLib EXPORT albany-export
    LIBRARY DESTINATION ${LIB_INSTALL_DIR}
    ARCHIVE DESTINATION ${LIB_INSTALL_DIR}) 
  FILE(WRITE "${CMAKE_BINARY_DIR}/albany-targets.cmake"
    "set(Albany_LIBRARIES ${ALBANY_LIBRARIES})")
  FILE(INSTALL "${CMAKE_BINARY_DIR}/albany-targets.cmake"
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${LIB_INSTALL_DIR}/Albany/cmake)
ENDIF()
