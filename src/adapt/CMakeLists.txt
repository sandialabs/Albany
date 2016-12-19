##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//


# Adaptation

SET(SOURCES
    AAdapt_AbstractAdapterT.cpp
    AAdapt_AdaptiveSolutionManagerT.cpp
    AAdapt_InitialCondition.cpp
    AAdapt_AnalyticFunction.cpp
    AAdapt_SolutionObserver.cpp
    AAdapt_RC_DataTypes.hpp
    AAdapt_RC_Manager.cpp
    AAdapt_RC_Field.cpp
    AAdapt_RC_Reader.cpp
    AAdapt_RC_Writer.cpp
    AAdapt_RC_Projector_impl.cpp
   )

SET(HEADERS
    AAdapt_AbstractAdapterT.hpp
    AAdapt_AdaptiveSolutionManagerT.hpp
    AAdapt_InitialCondition.hpp
    AAdapt_AnalyticFunction.hpp
    AAdapt_SolutionObserver.hpp
    AAdapt_RC_Manager.hpp
    AAdapt_RC_Field.hpp
    AAdapt_RC_Reader.hpp
    AAdapt_RC_Writer.hpp
    AAdapt_RC_Projector_impl.hpp
   )

IF(ALBANY_HAVE_STK)
  SET(SOURCES ${SOURCES}
    AAdapt_CopyRemeshT.cpp
  )
  SET(HEADERS ${HEADERS}
    AAdapt_CopyRemeshT.hpp
  )
ENDIF()


IF (ALBANY_EPETRA) 
  SET(SOURCES ${SOURCES} 
    AAdapt_AbstractAdapter.cpp
    AAdapt_AdaptationFactory.cpp
    AAdapt_AdaptiveSolutionManager.cpp
    AAdapt_AdaptiveModelFactory.cpp
    AAdapt_ThyraAdaptiveModelEvaluator.cpp
   ) 
  SET(HEADERS ${HEADERS} 
    AAdapt_AbstractAdapter.hpp
    AAdapt_AdaptationFactory.hpp
    AAdapt_AdaptiveSolutionManager.hpp
    AAdapt_AdaptiveModelFactory.hpp
    AAdapt_ThyraAdaptiveModelEvaluator.hpp
  )
  IF(ALBANY_HAVE_STK)
    SET(SOURCES ${SOURCES}
      AAdapt_CopyRemesh.cpp
    )
    SET(HEADERS ${HEADERS}
      AAdapt_CopyRemesh.hpp
    )
  ENDIF()
ENDIF()

# Random Criterion & Stress Fracture depend on LCM
# Speculative features and code, currently
IF(ALBANY_LCM AND LCM_SPECULATIVE AND ALBANY_HAVE_STK)

SET(SOURCES ${SOURCES}
    AAdapt_RandomCriterion.cpp
    AAdapt_StressFracture.cpp
   )

SET(HEADERS ${HEADERS}
    AAdapt_RandomCriterion.hpp
    AAdapt_StressFracture.hpp
   )

IF (ALBANY_EPETRA) 
SET(SOURCES ${SOURCES} 
    AAdapt_RandomFracture.cpp
    )
SET(HEADERS ${HEADERS}
    AAdapt_RandomFracture.hpp
    )
ENDIF()
ENDIF()

# Topology Modification depends on LCM and STK
IF(ALBANY_LCM AND ALBANY_HAVE_STK AND ALBANY_BGL)

SET(SOURCES ${SOURCES}
    AAdapt_TopologyModificationT.cpp
   )

SET(HEADERS ${HEADERS}
    AAdapt_TopologyModificationT.hpp
   )

IF (ALBANY_EPETRA)
 
SET(SOURCES ${SOURCES} 
    AAdapt_TopologyModification.cpp
    )
SET(HEADERS ${HEADERS}
    AAdapt_TopologyModification.hpp
    )
    
ENDIF()

ENDIF()

IF(ALBANY_LCM AND ALBANY_STK_PERCEPT AND ALBANY_HAVE_STK)

SET(SOURCES ${SOURCES}
    AAdapt_STKAdaptT.cpp
    AAdapt_STKUnifSizeField.cpp
   )

SET(HEADERS ${HEADERS}
    AAdapt_STKAdaptT.hpp
    AAdapt_STKAdaptT_Def.hpp
    AAdapt_STKUnifSizeField.hpp
   )

IF (ALBANY_EPETRA) 
SET(SOURCES ${SOURCES}
    AAdapt_STKAdapt.cpp
    )
SET(HEADERS ${HEADERS}
    AAdapt_STKAdapt.hpp
    AAdapt_STKAdapt_Def.hpp
)
ENDIF()

ENDIF()

# Mesh Adapt depends on SCOREC tools and PUMI
IF(ALBANY_SCOREC)

SET(SOURCES ${SOURCES}
    AAdapt_MeshAdapt.cpp
    AAdapt_MeshAdaptMethod.cpp
    AAdapt_UnifSizeField.cpp
    AAdapt_UnifRefSizeField.cpp
    AAdapt_NonUnifRefSizeField.cpp
    AAdapt_AlbanySizeField.cpp
    AAdapt_ExtrudedAdapt.cpp
   )

SET(HEADERS ${HEADERS}
    AAdapt_MeshAdaptMethod.hpp
    AAdapt_MeshAdapt.hpp
    AAdapt_UnifSizeField.hpp
    AAdapt_UnifRefSizeField.hpp
    AAdapt_NonUnifRefSizeField.hpp
    AAdapt_AlbanySizeField.hpp
    AAdapt_ExtrudedAdapt.hpp
   )

SET(SOURCES ${SOURCES} AAdapt_SPRSizeField.cpp)
SET(HEADERS ${HEADERS} AAdapt_SPRSizeField.hpp)

IF(ALBANY_OMEGA_H)
  SET(SOURCES ${SOURCES} AAdapt_Omega_h_Method.cpp)
  SET(HEADERS ${HEADERS} AAdapt_Omega_h_Method.hpp)
ENDIF()

include_directories(${ALBANY_PUMI_INCLUDE_DIRS} ${PUMI_INCLUDE_DIR})

ENDIF()

IF(ALBANY_AMP)
SET(SOURCES ${SOURCES} AAdapt_SimAdapt.cpp AAdapt_SimLayerAdapt.cpp)
SET(HEADERS ${HEADERS} AAdapt_SimAdapt.hpp AAdapt_SimLayerAdapt.hpp)
include_directories(${ALBANY_PUMI_INCLUDE_DIRS})
ENDIF()

include_directories(${Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
                    ${Albany_SOURCE_DIR} / src ${Albany_SOURCE_DIR} / src / evaluators
                    ${Albany_SOURCE_DIR} / src / problems ${Albany_SOURCE_DIR} / src / responses
                    ${Albany_SOURCE_DIR} / src / disc
                    ${Albany_SOURCE_DIR} / src / disc / stk ${Albany_SOURCE_DIR} / src / LCM /
                    ${Albany_SOURCE_DIR} / src / LCM / evaluators ${Albany_SOURCE_DIR} / src / LCM / utils
                    ${Albany_SOURCE_DIR}/src/AMP/responses
                   )

add_library(albanyAdapt ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})

set_target_properties(albanyAdapt PROPERTIES PUBLIC_HEADER "${HEADERS}")

target_link_libraries(albanyAdapt ${SCOREC_LIB})

IF(ALBANY_OMEGA_H)
  target_link_libraries(albanyAdapt Omega_h::omega_h)
ENDIF()

IF(INSTALL_ALBANY)
install(TARGETS albanyAdapt EXPORT albany-export
        LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
        ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
        PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()
