##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//


# Adaptation

SET(SOURCES
  Albany_AbstractAdapter.cpp
  Albany_AdaptationFactory.cpp
  Albany_CopyRemesh.cpp
  Albany_AdaptiveSolutionManager.cpp
  Albany_InitialCondition.cpp
  Albany_AnalyticFunction.cpp
  )

SET(HEADERS
  Albany_AbstractAdapter.hpp
  Albany_AdaptationFactory.hpp
  Albany_CopyRemesh.hpp
  Albany_AdaptiveSolutionManager.hpp
  Albany_InitialCondition.hpp
  Albany_AnalyticFunction.hpp
  )

# Topology Modification & Stress Fracture depend on LCM
# Speculative features and code, currently
IF (ALBANY_LCM AND LCM_SPECULATIVE)

  SET(SOURCES ${SOURCES}
    Albany_RandomFracture.cpp
    Albany_RandomCriterion.cpp
    Albany_TopologyModification.cpp
    Albany_StressFracture.cpp
    )

  SET(HEADERS ${HEADERS}
    Albany_RandomFracture.hpp
    Albany_RandomCriterion.hpp
    Albany_TopologyModification.hpp
    Albany_StressFracture.hpp
    )

ENDIF()

IF (ALBANY_LCM AND ALBANY_STK_PERCEPT)

  SET(SOURCES ${SOURCES}
	  Albany_STKAdapt.cpp
	  Albany_STKUnifSizeField.cpp
  )

  SET(HEADERS ${HEADERS}
	  Albany_STKAdapt.hpp
	  Albany_STKAdapt_Def.hpp
	  Albany_STKUnifSizeField.hpp
  )

ENDIF()

# Mesh Adapt depends on SCOREC tools and FMDB
IF (ALBANY_SCOREC)

  SET(SOURCES ${SOURCES}
	Albany_MeshAdapt.cpp
	Albany_UnifSizeField.cpp
	Albany_UnifRefSizeField.cpp
    )

  SET(HEADERS ${HEADERS}
	Albany_MeshAdapt.hpp
	Albany_MeshAdapt_Def.hpp
	Albany_UnifSizeField.hpp
	Albany_UnifRefSizeField.hpp
    )

  include_directories (${ALBANY_FMDB_INCLUDE_DIRS} ${FMDB_INCLUDE_DIR})

ENDIF()

include_directories (${Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
                     ${Albany_SOURCE_DIR}/src ${Albany_SOURCE_DIR}/src/evaluators
                     ${Albany_SOURCE_DIR}/src/problems ${Albany_SOURCE_DIR}/src/responses
                     ${Albany_SOURCE_DIR}/src/disc
                     ${Albany_SOURCE_DIR}/src/disc/stk ${Albany_SOURCE_DIR}/src/LCM/
                     ${Albany_SOURCE_DIR}/src/LCM/evaluators ${Albany_SOURCE_DIR}/src/LCM/utils
                    )

add_library(albanyAdapt ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})

set_target_properties(albanyAdapt PROPERTIES PUBLIC_HEADER "${HEADERS}")

IF (INSTALL_ALBANY)
  install(TARGETS albanyAdapt EXPORT albany-export
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()
