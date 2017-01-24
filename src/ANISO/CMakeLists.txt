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
${ANISO_DIR}/ANISO_Expression.cpp
${ANISO_DIR}/ANISO_Time.cpp
${ANISO_DIR}/AdvectionKappa.cpp
${ANISO_DIR}/AdvectionAlpha.cpp
${ANISO_DIR}/AdvectionTau.cpp
${ANISO_DIR}/AdvectionResidual.cpp
)
set(aniso-evaluator-headers
${ANISO_DIR}/ANISO_Expression.hpp
${ANISO_DIR}/ANISO_Time.hpp
${ANISO_DIR}/ANISO_Time_Def.hpp
${ANISO_DIR}/AdvectionKappa.hpp
${ANISO_DIR}/AdvectionKappa_Def.hpp
${ANISO_DIR}/AdvectionAlpha.hpp
${ANISO_DIR}/AdvectionAlpha_Def.hpp
${ANISO_DIR}/AdvectionTau.hpp
${ANISO_DIR}/AdvectionTau_Def.hpp
${ANISO_DIR}/AdvectionResidual.hpp
${ANISO_DIR}/AdvectionResidual_Def.hpp
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
