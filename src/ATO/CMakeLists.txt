##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

SET(SOURCES
  ${CMAKE_SOURCE_DIR}/src/ATO/ATO_Solver.cpp
  ${CMAKE_SOURCE_DIR}/src/ATO/evaluators/ATO_StiffnessObjective.cpp
  ${CMAKE_SOURCE_DIR}/src/ATO/evaluators/ATO_Stress.cpp
  ${CMAKE_SOURCE_DIR}/src/ATO/evaluators/ATO_TopologyWeighting.cpp
  ${CMAKE_SOURCE_DIR}/src/ATO/problems/LinearElasticityProblem.cpp
  ${CMAKE_SOURCE_DIR}/src/ATO/problems/ATO_OptimizationProblem.cpp
  ${CMAKE_SOURCE_DIR}/src/ATO/utils/ATO_TopoTools.cpp
  ${CMAKE_SOURCE_DIR}/src/ATO/utils/ATO_Aggregator.cpp
  ${CMAKE_SOURCE_DIR}/src/ATO/utils/ATO_Optimizer.cpp
)

SET(HEADERS
  ${CMAKE_SOURCE_DIR}/src/ATO/ATO_Solver.hpp
  ${CMAKE_SOURCE_DIR}/src/ATO/evaluators/ATO_StiffnessObjective_Def.hpp
  ${CMAKE_SOURCE_DIR}/src/ATO/evaluators/ATO_StiffnessObjective.hpp
  ${CMAKE_SOURCE_DIR}/src/ATO/evaluators/ATO_Stress_Def.hpp
  ${CMAKE_SOURCE_DIR}/src/ATO/evaluators/ATO_Stress.hpp
  ${CMAKE_SOURCE_DIR}/src/ATO/evaluators/ATO_TopologyWeighting_Def.hpp
  ${CMAKE_SOURCE_DIR}/src/ATO/evaluators/ATO_TopologyWeighting.hpp
  ${CMAKE_SOURCE_DIR}/src/ATO/problems/LinearElasticityProblem.hpp
  ${CMAKE_SOURCE_DIR}/src/ATO/problems/ATO_OptimizationProblem.hpp
  ${CMAKE_SOURCE_DIR}/src/ATO/utils/ATO_TopoTools.hpp
  ${CMAKE_SOURCE_DIR}/src/ATO/utils/ATO_Aggregator.hpp
  ${CMAKE_SOURCE_DIR}/src/ATO/utils/ATO_Optimizer.hpp
)

include_directories (${Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/src 
  ${CMAKE_SOURCE_DIR}/src/evaluators
  ${CMAKE_SOURCE_DIR}/src/problems 
  ${CMAKE_SOURCE_DIR}/src/responses
  ${CMAKE_SOURCE_DIR}/src/LCM/
  ${CMAKE_SOURCE_DIR}/src/LCM/evaluators 
  ${CMAKE_SOURCE_DIR}/src/LCM/utils
  ${CMAKE_SOURCE_DIR}/src/ATO/
  ${CMAKE_SOURCE_DIR}/src/ATO/evaluators 
  ${CMAKE_SOURCE_DIR}/src/ATO/problems
  ${CMAKE_SOURCE_DIR}/src/ATO/utils
)

#message("checking for NLOPT")
#IF ( NLOPT )
#  message("adding NLOPT")
#  message("added ${NLOPT_INCLUDE_DIR} to include directories")
#  include_directories (${NLOPT_INCLUDE_DIR})
#  link_directories (${NLOPT_LIBRARY_DIR})
#  set(ALBANY_LIBRARIES ${ALBANY_LIBRARIES} nlopt)
#ENDIF()


add_library(ATO ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})

IF ( INSTALL_ALBANY )
    install(TARGETS ATO EXPORT Albany
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()

