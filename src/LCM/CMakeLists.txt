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
file (GLOB evaluator-sources
  "${LCM_DIR}/evaluators/*.cpp"
)
file (GLOB evaluator-headers
  "${LCM_DIR}/evaluators/*.hpp"
  "${LCM_DIR}/evaluators/*_Def.hpp"
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
  ${evaluator-sources}
)

set(HEADERS ${HEADERS}
  ${problem-headers}
  ${evaluator-headers}
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
