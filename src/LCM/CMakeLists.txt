##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//
#
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
  "${LCM_DIR}/parallel_evaluators/*.cpp"
)
file (GLOB evaluator-headers
  "${LCM_DIR}/evaluators/*.hpp"
  "${LCM_DIR}/parallel_evaluators/*.hpp"
)

#LCM model evaluators 
file (GLOB model-eval-sources
  "${LCM_DIR}/*.cpp"
)
file (GLOB model-eval-headers
  "${LCM_DIR}/*.hpp"
)

# LCM surface element evaluators
file (GLOB surface-element-sources
  "${LCM_DIR}/evaluators/surface-element/*.cpp"
)
file (GLOB surface-element-headers
  "${LCM_DIR}/evaluators/surface-element/*.hpp"
)

# LCM special boundary conditions
file (GLOB bc-sources
  "${LCM_DIR}/evaluators/bc/*.cpp"
)
file (GLOB bc-headers
  "${LCM_DIR}/evaluators/bc/*.hpp"
)

#LCM models
file (GLOB models-sources
  "${LCM_DIR}/models/*.cpp"
  "${LCM_DIR}/parallel_models/*.cpp"
)
file (GLOB models-headers
  "${LCM_DIR}/models/*.hpp"
  "${LCM_DIR}/models/core/CrystalPlasticity/*.hpp"
  "${LCM_DIR}/parallel_models/*.hpp"
)

IF(ALBANY_HAVE_STK)
#LCM utils
file (GLOB utils-sources
  "${LCM_DIR}/utils/*.cpp"
  "${LCM_DIR}/utils/*.cc"
)
file (GLOB utils-headers
  "${LCM_DIR}/utils/*.hpp"
  "${LCM_DIR}/utils/*.h"
)

#LCM utils/topology
  IF(ALBANY_BGL)
  file (GLOB topology-sources
    "${LCM_DIR}/utils/topology/*.cc"
  )
  file (GLOB topology-headers
    "${LCM_DIR}/utils/topology/*.h"
  )
  ENDIF(ALBANY_BGL)

ELSE()
  set(utils-headers ${utils-headers}
    ${LCM_DIR}/utils/LCM_Utils.h
    ${LCM_DIR}/utils/Projection.hpp
  )
  set(utils-sources ${utils-sources}
    ${LCM_DIR}/utils/LCM_Utils.cc
    ${LCM_DIR}/utils/Projection.cpp
  )
ENDIF()

set(SOURCES ${SOURCES}
  ${problem-sources}
  ${evaluator-sources}
  ${model-eval-sources}
  ${surface-element-sources}
  ${bc-sources}
  ${models-sources}
  ${utils-sources}
  ${topology-sources}
  )
set(HEADERS ${HEADERS}
  ${problem-headers}
  ${evaluator-headers}
  ${model-eval-headers}
  ${surface-element-headers}
  ${bc-headers}
  ${models-headers}
  ${utils-headers}
  ${topology-headers}
 )

IF(NOT ALBANY_PERIDIGM OR NOT ALBANY_EPETRA)
  list(REMOVE_ITEM SOURCES ${LCM_DIR}/problems/PeridigmProblem.cpp)
  list(REMOVE_ITEM SOURCES ${LCM_DIR}/utils/PeridigmManager.cpp)
  list(REMOVE_ITEM SOURCES ${LCM_DIR}/evaluators/PeridigmForce.cpp)
  list(REMOVE_ITEM SOURCES ${LCM_DIR}/evaluators/PeridigmPartialStress.cpp)
  list(REMOVE_ITEM SOURCES ${LCM_DIR}/evaluators/AlbanyPeridigmOBCFunctional.cpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/problems/PeridigmProblem.hpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/utils/PeridigmManager.hpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/evaluators/PeridigmForce_Def.hpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/evaluators/PeridigmForce.hpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/evaluators/PeridigmPartialStress_Def.hpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/evaluators/PeridigmPartialStress.hpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/evaluators/AlbanyPeridigmOBCFunctional_Def.hpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/evaluators/AlbanyPeridigmOBCFunctional.hpp)
ENDIF()

IF(NOT ALBANY_CONTACT)
  list(REMOVE_ITEM SOURCES ${LCM_DIR}/evaluators/MortarContactResidual.cpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/evaluators/MortarContactResidual.hpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/evaluators/MortarContactResidual_Def.hpp)
  list(REMOVE_ITEM SOURCES ${LCM_DIR}/evaluators/MortarContactConstraints.cpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/evaluators/MortarContactConstraints.hpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/evaluators/MortarContactConstraints_Def.hpp)
ENDIF()

IF(NOT ALBANY_HAVE_STK)
  list(REMOVE_ITEM SOURCES ${LCM_DIR}/Schwarz_BoundaryJacobian.cpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/Schwarz_BoundaryJacobian.hpp)
  list(REMOVE_ITEM SOURCES ${LCM_DIR}/Schwarz_CoupledJacobian.cpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/Schwarz_CoupledJacobian.hpp)
  list(REMOVE_ITEM SOURCES ${LCM_DIR}/SchwarzMultiscale.cpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/SchwarzMultiscale.hpp)
  list(REMOVE_ITEM SOURCES ${LCM_DIR}/evaluators/bc/SchwarzBC.cpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/evaluators/bc/SchwarzBC_Def.hpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/evaluators/bc/SchwarzBC.hpp)
  list(REMOVE_ITEM SOURCES ${LCM_DIR}/evaluators/bc/PDNeighborFitBC.cpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/evaluators/bc/PDNeighborFitBC_Def.hpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/evaluators/bc/PDNeighborFitBC.hpp)
ENDIF()

IF(Kokkos_ENABLE_Cuda)
  list(REMOVE_ITEM SOURCES ${LCM_DIR}/models/GursonModel.cpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/models/GursonModel.hpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/models/GursonModel_Def.hpp)
  list(REMOVE_ITEM SOURCES ${LCM_DIR}/models/GursonHMRModel.cpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/models/GursonHMRModel.hpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/models/GursonHMRModel_Def.hpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/models/FM_AlbanyInterface.hpp)
  list(REMOVE_ITEM SOURCES ${LCM_DIR}/models/FM_AlbanyInterface.cpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/models/FM_AlbanyInterface_Def.hpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/models/FerroicCore.cpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/models/FerroicCore.hpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/models/FerroicCore_Def.hpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/models/FerroicModel.hpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/models/FerroicModel_Def.hpp)
  list(REMOVE_ITEM SOURCES ${LCM_DIR}/models/CrystalPlasticityCore.cpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/models/CrystalPlasticityCore.hpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/models/CrystalPlasticityCore_Def.hpp)
  list(REMOVE_ITEM SOURCES ${LCM_DIR}/models/CrystalPlasticityModel.cpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/models/CrystalPlasticityModel.hpp)
  list(REMOVE_ITEM HEADERS ${LCM_DIR}/models/CrystalPlasticityModel_Def.hpp)
ENDIF()

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
  ${LCM_DIR}/evaluators/surface-element
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

IF ( INSTALL_ALBANY )
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
  add_executable(MaterialPointSimulator test/utils/MaterialPointSimulator.cc)
  add_executable(MeshComponents test/utils/MeshComponents.cc)
  add_executable(NodeUpdate test/utils/NodeUpdate.cc)
  add_executable(PartitionTest test/utils/PartitionTest.cc)
  add_executable(BifurcationTest test/utils/BifurcationTest.cc)
  add_executable(Subdivision test/utils/Subdivision.cc)
  add_executable(MinSurfaceMPS test/utils/MinSurfaceMPS.cc)
  add_executable(MinSurfaceOutput test/utils/MinSurfaceOutput.cc)
  add_executable(BoundarySurfaceOutput test/utils/BoundarySurfaceOutput.cc)
  add_executable(Test1_Subdivision test/utils/Test1_Subdivision.cc)
  add_executable(Test2_Subdivision test/utils/Test2_Subdivision.cc)
  add_executable(TopologyBase test/utils/TopologyBase.cc)

  add_executable(
    utLocalNonlinearSolver
    test/unit_tests/StandardUnitTestMain.cpp
    test/unit_tests/utLocalNonlinearSolver.cpp
    )

  add_executable(
    utMiniSolvers
    test/unit_tests/utMiniSolvers.cc
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
    SET(LCM_UT_LIBS albanyLib albanyAdapt pumi albanySTK LCM albanyLib)
  ELSE()
    SET(LCM_UT_LIBS albanyLib albanyAdapt albanySTK LCM albanyLib)
  ENDIF()
ELSE(BUILD_SHARED_LIBS)
  # We really don't link to anything in Hydride
  IF(ALBANY_SCOREC)
    SET(LCM_UT_LIBS LCM albanyAdapt albanyLib pumi albanySTK albanyLib)
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
  target_link_libraries(MaterialPointSimulator ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(MeshComponents ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(NodeUpdate ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(PartitionTest ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(BifurcationTest ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(Subdivision ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(MinSurfaceMPS ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(MinSurfaceOutput ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(BoundarySurfaceOutput ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(Test1_Subdivision ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(Test2_Subdivision ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(TopologyBase ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(utLocalNonlinearSolver ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(utMiniSolvers ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(utSurfaceElement ${repeat_libs} ${ALL_LIBRARIES})
  target_link_libraries(utHeliumODEs ${repeat_libs} ${ALL_LIBRARIES})
ENDIF()
