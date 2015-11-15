##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# a posteriori goal-oriented error estimation

SET(GOAL_DIR ${Albany_SOURCE_DIR}/src/GOAL)
SET(LCM_DIR ${Albany_SOURCE_DIR}/src/LCM)

SET(SOURCES
  utils/GOAL_BCUtils.cpp
  evaluators/GOAL_ComputeHierarchicBasis.cpp
  evaluators/GOAL_LpStress.cpp
  evaluators/GOAL_ScatterQoI.cpp
  problems/GOAL_MechanicsProblem.cpp
  responses/GOAL_AdjointResponse.cpp)

SET(HEADERS
  utils/GOAL_BCUtils.hpp
  evaluators/GOAL_ComputeHierarchicBasis.hpp
  evaluators/GOAL_ComputeHierarchicBasis_Def.hpp
  evaluators/GOAL_LpStress.hpp
  evaluators/GOAL_LpStress_Def.hpp
  evaluators/GOAL_ScatterQoI.hpp
  evaluators/GOAL_ScatterQoI_Def.hpp
  problems/GOAL_MechanicsProblem.hpp
  responses/GOAL_AdjointResponse.hpp)

include_directories(
  ${Trilinos_INCLUDE_DIRS}
  ${Trilinos_TPL_INCLUDE_DIRS}
  ${LCM_DIR}/utils
  ${LCM_DIR}/models
  ${LCM_DIR}/problems
  ${LCM_DIR}/evaluators
  ${GOAL_DIR}/problems)

add_library(GOAL ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})

set_target_properties(GOAL PROPERTIES PUBLIC_HEADER "${HEADERS}")

IF(INSTALL_ALBANY)
install(TARGETS GOAL EXPORT albany-export
        LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
        ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
        PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()
