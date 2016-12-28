##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

SET(ANISO_DIR ${Albany_SOURCE_DIR}/src/ANISO)

set(aniso-problem-sources
${ANISO_DIR}/AdvectionProblem.cpp
)
set(aniso-problem-headers
${ANISO_DIR}/AdvectionProblem.hpp
)

set(aniso-evaluator-sources
)
set(aniso-evaluator-headers
)

set(aniso-response-sources
)
set(aniso-response-headers
)

set(SOURCES
  ${aniso-problem-sources}
  ${aniso-evaluator-sources}
  ${aniso-response-sources}
)

set(HEADERS
  ${aniso-problem-headers}
  ${aniso-evaluator-headers}
  ${aniso-response-headers}
)

set(INCLUDE_DIRS
  ${Trilinos_INCLUDE_DIRS}
  ${Albany_SOURCE_DIR}/src
  ${Albany_SOURCE_DIR}/src/evaluators
  ${Albany_SOURCE_DIR}/src/problems 
  ${Albany_SOURCE_DIR}/src/responses
  ${Trilinos_TPL_INCLUDE_DIRS}
)

include_directories(${INCLUDE_DIRS})
link_directories (${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})
add_library(ANISO ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})
set_target_properties(ANISO PROPERTIES PUBLIC_HEADER "${HEADERS}")

IF (INSTALL_ALBANY)
  install(TARGETS ANISO EXPORT albany-export
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()
