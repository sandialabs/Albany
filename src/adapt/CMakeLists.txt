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
  )

SET(HEADERS
	Albany_AbstractAdapter.hpp
	Albany_AdaptationFactory.hpp 
	Albany_CopyRemesh.hpp
	Albany_AdaptiveSolutionManager.hpp
  )

# Topology Modification & Stress Fracture depend on LCM
IF (ALBANY_LCM)

SET(SOURCES ${SOURCES}
	Albany_TopologyModification.cpp
	Albany_StressFracture.cpp
  )

SET(HEADERS ${HEADERS}
	Albany_TopologyModification.hpp
	Albany_StressFracture.hpp
  )

ENDIF()

# Mesh Adapt depends on SCOREC tools and FMDB
IF (ALBANY_SCOREC)

SET(SOURCES ${SOURCES}
	Albany_MeshAdapt.cpp
  )

SET(HEADERS ${HEADERS}
	Albany_MeshAdapt.hpp
  )

ENDIF()

include_directories (${Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
                     ${Albany_SOURCE_DIR}/src ${Albany_SOURCE_DIR}/src/evaluators 
                     ${Albany_SOURCE_DIR}/src/problems ${Albany_SOURCE_DIR}/src/responses 
                     ${Albany_SOURCE_DIR}/src/stk ${Albany_SOURCE_DIR}/src/LCM/ 
                     ${Albany_SOURCE_DIR}/src/LCM/evaluators ${Albany_SOURCE_DIR}/src/LCM/utils
                    )

add_library(adapt ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})
