##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//
# AFRL

# problems
SET(SOURCES
  problems/MultiScaleHeatProblem.cpp
  problems/RotationMechanicsProblem.cpp
  )
SET(HEADERS
  problems/MultiScaleHeatProblem.hpp
  problems/RotationMechanicsProblem.hpp
  )

# evaluators
SET(SOURCES ${SOURCES}
  evaluators/MultiScaleThermalConductivity.cpp
  evaluators/RotatingReferenceFrame.cpp
  )

SET(HEADERS ${HEADERS}
  evaluators/MultiScaleThermalConductivity.hpp
  evaluators/MultiScaleThermalConductivity_Def.hpp
  evaluators/RotatingReferenceFrame.hpp
  evaluators/RotatingReferenceFrame_Def.hpp
  )

# communicators
SET(SOURCES ${SOURCES}
  communicators/RPCFunctor.cpp
  )

SET(HEADERS ${HEADERS}
  communicators/RPCFunctor.hpp
  )

include_directories (${Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
  ${Albany_SOURCE_DIR}/src ${Albany_SOURCE_DIR}/src/evaluators
  ${Albany_SOURCE_DIR}/src/problems ${Albany_SOURCE_DIR}/src/responses
  ${Albany_SOURCE_DIR}/src/stk ${Albany_SOURCE_DIR}/src/LCM/problems
  ${Albany_SOURCE_DIR}/src/LCM/evaluators
  ./problems ./evaluators ./communicators
  )

add_library(AFRL ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})

IF (AFRL_USE_RABBITMQ)
  set_target_properties(AFRL PROPERTIES COMPILE_DEFINITIONS "USE_RABBITMQ")
ENDIF()
target_include_directories(AFRL PUBLIC ${ZeroMQ_INCLUDE_DIR}
  ${RabbitMQ_INCLUDE_DIR} ${OPENSSL_INCLUDE_DIR})
target_link_libraries(AFRL LINK_PUBLIC ${ZeroMQ_LIBRARIES}
  ${RabbitMQ_LIBRARIES} ${OPENSSL_LIBRARIES})

IF ( INSTALL_ALBANY )
    install(TARGETS AFRL EXPORT Albany
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()
