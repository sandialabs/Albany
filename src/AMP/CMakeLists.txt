##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

SET(AMP_DIR ${Albany_SOURCE_DIR}/src/AMP)

set(amp-problem-sources
  "${AMP_DIR}/problems/AMPThermoMechanics.cpp"
  "${AMP_DIR}/problems/PhaseProblem.cpp"
)
set(amp-problem-headers
  "${AMP_DIR}/problems/AMPThermoMechanics.hpp"
  "${AMP_DIR}/problems/PhaseProblem.hpp"
)

set(amp-evaluator-sources
  "${AMP_DIR}/evaluators/AMP_Time.cpp"
  "${AMP_DIR}/evaluators/EnergyDot.cpp"
  "${AMP_DIR}/evaluators/Laser.cpp"
  "${AMP_DIR}/evaluators/LaserSource.cpp"
  "${AMP_DIR}/evaluators/Local_Porosity.cpp"
  "${AMP_DIR}/evaluators/PhaseResidual.cpp"
  "${AMP_DIR}/evaluators/PhaseSource.cpp"
  "${AMP_DIR}/evaluators/Phi.cpp"
  "${AMP_DIR}/evaluators/Psi.cpp"
  "${AMP_DIR}/evaluators/RhoCp.cpp"
  "${AMP_DIR}/evaluators/ThermalCond.cpp"
)
set(amp-evaluator-headers
  "${AMP_DIR}/evaluators/AMP_Time_Def.hpp"
  "${AMP_DIR}/evaluators/AMP_Time.hpp"
  "${AMP_DIR}/evaluators/EnergyDot_Def.hpp"
  "${AMP_DIR}/evaluators/EnergyDot.hpp"
  "${AMP_DIR}/evaluators/Laser.hpp"
  "${AMP_DIR}/evaluators/LaserSource_Def.hpp"
  "${AMP_DIR}/evaluators/LaserSource.hpp"
  "${AMP_DIR}/evaluators/Local_Porosity_Def.hpp"
  "${AMP_DIR}/evaluators/Local_Porosity.hpp"
  "${AMP_DIR}/evaluators/PhaseResidual_Def.hpp"
  "${AMP_DIR}/evaluators/PhaseResidual.hpp"
  "${AMP_DIR}/evaluators/PhaseSource_Def.hpp"
  "${AMP_DIR}/evaluators/PhaseSource.hpp"
  "${AMP_DIR}/evaluators/Phi_Def.hpp"
  "${AMP_DIR}/evaluators/Phi.hpp"
  "${AMP_DIR}/evaluators/Psi_Def.hpp"
  "${AMP_DIR}/evaluators/Psi.hpp"
  "${AMP_DIR}/evaluators/RhoCp_Def.hpp"
  "${AMP_DIR}/evaluators/RhoCp.hpp"
  "${AMP_DIR}/evaluators/ThermalCond_Def.hpp"
  "${AMP_DIR}/evaluators/ThermalCond.hpp"
)

set(amp-response-sources
  "${AMP_DIR}/responses/Energy.cpp"
  "${AMP_DIR}/responses/EnergyIntegral.cpp"
  "${AMP_DIR}/responses/PathSizeField.cpp"
)
set(amp-response-headers
  "${AMP_DIR}/responses/Energy_Def.hpp"
  "${AMP_DIR}/responses/Energy.hpp"
  "${AMP_DIR}/responses/EnergyIntegral.hpp"
  "${AMP_DIR}/responses/PathSizeField.hpp"
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
