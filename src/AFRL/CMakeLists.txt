##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//
# AFRL

# problems
SET(SOURCES
  problems/MultiScaleHeatProblem.cpp
  )
SET(HEADERS
  problems/MultiScaleHeatProblem.hpp
  )

# evaluators
SET(SOURCES ${SOURCES}
  evaluators/MultiScaleThermalConductivity.cpp
  )

SET(HEADERS ${HEADERS}
  evaluators/MultiScaleThermalConductivity.hpp
  evaluators/MultiScaleThermalConductivity_Def.hpp
  )

include_directories (${Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
  ${Albany_SOURCE_DIR}/src ${Albany_SOURCE_DIR}/src/evaluators
  ${Albany_SOURCE_DIR}/src/problems ${Albany_SOURCE_DIR}/src/responses
  ${Albany_SOURCE_DIR}/src/stk
  ./problems ./evaluators
  )

add_library(AFRL ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})
target_link_libraries(AFRL LINK_PUBLIC ${ZeroMQ_LIBRARIES})

IF ( INSTALL_ALBANY )
    install(TARGETS AFRL EXPORT Albany
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()
