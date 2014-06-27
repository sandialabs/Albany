##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

SET(SEE_DIR ${Albany_SOURCE_DIR}/src/SEE)

file (GLOB amp-problem-sources
  "${SEE_DIR}/problems/*.cpp"
)
file (GLOB amp-problem-headers
  "${SEE_DIR}/problems/*.hpp"
)

file (GLOB amp-evaluator-sources
  "${SEE_DIR}/evaluators/*.cpp"
)
file (GLOB amp-evaluator-headers
  "${SEE_DIR}/evaluators/*.hpp"
)

set(SOURCES
  ${amp-problem-sources}
  ${amp-evaluator-sources}
  )

set(HEADERS
  ${amp-problem-headers}
  ${amp-evaluator-headers}
 )

set(INCLUDE_DIRS
  ${Trilinos_INCLUDE_DIRS}
  ${Albany_SOURCE_DIR}/src
  ${Albany_SOURCE_DIR}/src/evaluators
  ${Albany_SOURCE_DIR}/src/problems 
  ${Albany_SOURCE_DIR}/src/responses
  ${Albany_SOURCE_DIR}/src/disc/stk
  ${Albany_SOURCE_DIR}/src/adapt
  ${SEE_DIR}/problems
  ${SEE_DIR}/evaluators
  ${SEE_DIR}/evaluators/bc
  ${SEE_DIR}/evaluators/surface-element
  ${SEE_DIR}/models
  ${SEE_DIR}/utils
  ${SEE_DIR}/utils/topology
  ${ALBANY_FMDB_INCLUDE_DIRS} 
  ${Trilinos_TPL_INCLUDE_DIRS}
)

include_directories(${INCLUDE_DIRS})
link_directories (${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})
add_library(SEE ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})
set_target_properties(SEE PROPERTIES PUBLIC_HEADER "${HEADERS}")

IF ( INSTALL_ALBANY )
  install(TARGETS SEE EXPORT albany-export
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()
