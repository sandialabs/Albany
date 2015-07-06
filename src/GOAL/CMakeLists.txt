##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# a posteriori goal-oriented error estimation

SET(GOAL_DIR ${Albany_SOURCE_DIR}/src/GOAL)

SET(SOURCES
  GOAL_BCManager.cpp
  GOAL_Discretization.cpp
  GOAL_FieldManagerBundle.cpp
  GOAL_LinearSystem.cpp
  evaluators/GOAL_MechanicsAdjoint.cpp
  responses/GOAL_MechAdjResponse.cpp)

SET(HEADERS
  GOAL_BCManager.hpp
  GOAL_Discretization.hpp
  GOAL_FieldManagerBundle.hpp
  GOAL_LinearSystem.hpp
  evaluators/GOAL_MechanicsAdjoint.hpp
  evaluators/GOAL_MechanicsAdjoint_Def.hpp
  responses/GOAL_MechAdjResponse.hpp)

include_directories(
  ${Trilinos_INCLUDE_DIRS}
  ${Trilinos_TPL_INCLUDE_DIRS}
  ${Albany_SOURCE_DIR}/LCM/problems
  ${GOAL_DIR}/responses
  ${GOAL_DIR}/evaluators)

add_library(GOAL ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})

set_target_properties(GOAL PROPERTIES PUBLIC_HEADER "${HEADERS}")

IF(INSTALL_ALBANY)
install(TARGETS GOAL EXPORT albany-export
        LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
        ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
        PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()
