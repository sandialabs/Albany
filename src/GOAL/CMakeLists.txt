##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# a posteriori goal-oriented error estimation

if(ALBANY_GOAL_EXPERIMENTAL)

  set(GOAL_GIT_CMD "git")
  set(GOAL_GIT_ARG1 "clone")
  set(GOAL_GIT_ARG2 "https://github.com/bgranzow/goal.git")
  if(NOT IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/goal)
    execute_process(
      COMMAND ${GOAL_GIT_CMD} ${GOAL_GIT_ARG1} ${GOAL_GIT_ARG2}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

endif()
