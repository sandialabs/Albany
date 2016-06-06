##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# a posteriori goal-oriented error estimation

SET(LCM_DIR ${Albany_SOURCE_DIR}/src/LCM)
SET(DISC_DIR ${Albany_SOURCE_DIR}/src/disc)

SET(SOURCES
GOAL_Main.cpp
)

SET(HEADERS
)

include_directories(
${Trilinos_INCLUDE_DIRS}
${Trilinos_TPL_INCLUDE_DIRS}
${LCM_DIR}/utils
${LCM_DIR}/models
${LCM_DIR}/evaluators
${GOAL_DIR}
)

add_library(GOAL ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})
set_target_properties(GOAL PROPERTIES PUBLIC_HEADER "${HEADERS}")

IF(INSTALL_ALBANY)
install(TARGETS GOAL EXPORT albany-export
        LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
        ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
        PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()
