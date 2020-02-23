##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

SET(TDM_DIR ${Albany_SOURCE_DIR}/src/TDM)

set(TDM-problem-sources
  "${TDM_DIR}/problems/TDManufacturing.cpp"
)
set(TDM-problem-headers
  "${TDM_DIR}/problems/TDManufacturing.hpp"
)

set(TDM-evaluator-sources
  "${TDM_DIR}/evaluators/TDM_Time.cpp"
  "${TDM_DIR}/evaluators/Phi1.cpp"
  "${TDM_DIR}/evaluators/Psi1.cpp"
  "${TDM_DIR}/evaluators/Depth.cpp"
  "${TDM_DIR}/evaluators/Phi2.cpp"
  "${TDM_DIR}/evaluators/Psi2.cpp"
  "${TDM_DIR}/evaluators/ThCond.cpp"
  "${TDM_DIR}/evaluators/rho_Cp.cpp"
  "${TDM_DIR}/evaluators/Energy_Dot.cpp"
  "${TDM_DIR}/evaluators/TDM_Residual.cpp"
  "${TDM_DIR}/evaluators/Laser_Source.cpp"
  "${TDM_DIR}/evaluators/Laser.cpp"
)
set(TDM-evaluator-headers
  "${TDM_DIR}/evaluators/TDM_Time_Def.hpp"
  "${TDM_DIR}/evaluators/TDM_Time.hpp"
  "${TDM_DIR}/evaluators/Phi1_Def.hpp"
  "${TDM_DIR}/evaluators/Phi1.hpp"
  "${TDM_DIR}/evaluators/Psi1_Def.hpp"
  "${TDM_DIR}/evaluators/Psi1.hpp"
  "${TDM_DIR}/evaluators/Phi2_Def.hpp"
  "${TDM_DIR}/evaluators/Phi2.hpp"
  "${TDM_DIR}/evaluators/Psi2_Def.hpp"
  "${TDM_DIR}/evaluators/Psi2.hpp"
  "${TDM_DIR}/evaluators/Depth_Def.hpp"
  "${TDM_DIR}/evaluators/Depth.hpp"
  "${TDM_DIR}/evaluators/ThCond_Def.hpp"
  "${TDM_DIR}/evaluators/ThCond.hpp"
  "${TDM_DIR}/evaluators/rho_Cp_Def.hpp"
  "${TDM_DIR}/evaluators/rho_Cp.hpp"
  "${TDM_DIR}/evaluators/Energy_Dot_Def.hpp"
  "${TDM_DIR}/evaluators/Energy_Dot.hpp"
  "${TDM_DIR}/evaluators/TDM_Residual_Def.hpp"
  "${TDM_DIR}/evaluators/TDM_Residual.hpp"
  "${TDM_DIR}/evaluators/Laser_Source_Def.hpp"
  "${TDM_DIR}/evaluators/Laser_Source.hpp"
  "${TDM_DIR}/evaluators/Laser.hpp"
  "${TDM_DIR}/evaluators/SurfaceUpdate.hpp"  
)

set(SOURCES
  ${TDM-problem-sources}
  ${TDM-evaluator-sources}
  )

set(HEADERS
  ${TDM-problem-headers}
  ${TDM-evaluator-headers}
 )

set(INCLUDE_DIRS
  ${Trilinos_INCLUDE_DIRS}
  ${Albany_SOURCE_DIR}/src
  ${Albany_SOURCE_DIR}/src/evaluators
  ${Albany_SOURCE_DIR}/src/problems 
  ${Albany_SOURCE_DIR}/src/disc/pumi
  ${Albany_SOURCE_DIR}/src/adapt
  ${TDM_DIR}/problems
  ${TDM_DIR}/evaluators
  ${ALBANY_PUMI_INCLUDE_DIRS} 
  ${Trilinos_TPL_INCLUDE_DIRS}
)

include_directories(${INCLUDE_DIRS})
link_directories (${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})
add_library(TDM ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})
set_target_properties(TDM PROPERTIES PUBLIC_HEADER "${HEADERS}")

IF ( INSTALL_ALBANY )
  install(TARGETS TDM EXPORT albany-export
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()
