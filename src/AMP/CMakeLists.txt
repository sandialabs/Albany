##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

SET(AMP_DIR ${Albany_SOURCE_DIR}/src/AMP)

file (GLOB amp-problem-sources
  "${AMP_DIR}/problems/*.cpp"
)
file (GLOB amp-problem-headers
  "${AMP_DIR}/problems/*.hpp"
)

file (GLOB amp-evaluator-sources
  "${AMP_DIR}/evaluators/*.cpp"
)
file (GLOB amp-evaluator-headers
  "${AMP_DIR}/evaluators/*.hpp"
)

file (GLOB amp-response-sources
  "${AMP_DIR}/responses/*.cpp"
)
file (GLOB amp-response-headers
  "${AMP_DIR}/responses/*.hpp"
)

set(SOURCES
  ${amp-problem-sources}
  ${amp-evaluator-sources}
  ${amp-response-sources}
  )

set(HEADERS
  ${amp-problem-headers}
  ${amp-evaluator-headers}
  ${amp-response-headers}
 )

set(INCLUDE_DIRS
  ${Trilinos_INCLUDE_DIRS}
  ${Albany_SOURCE_DIR}/src
  ${Albany_SOURCE_DIR}/src/evaluators
  ${Albany_SOURCE_DIR}/src/problems 
  ${Albany_SOURCE_DIR}/src/responses
  ${Albany_SOURCE_DIR}/src/disc/pumi
  ${Albany_SOURCE_DIR}/src/adapt
  ${AMP_DIR}/problems
  ${AMP_DIR}/evaluators
  ${ALBANY_PUMI_INCLUDE_DIRS} 
  ${Trilinos_TPL_INCLUDE_DIRS}
)

include_directories(${INCLUDE_DIRS})
link_directories (${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})
add_library(AMP ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})
set_target_properties(AMP PROPERTIES PUBLIC_HEADER "${HEADERS}")

IF ( INSTALL_ALBANY )
  install(TARGETS AMP EXPORT albany-export
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()
