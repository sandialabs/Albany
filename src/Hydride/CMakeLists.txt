##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//
# HYDRIDE

# problems
SET(SOURCES
  problems/HydrideProblem.cpp
  problems/HydMorphProblem.cpp
  problems/LaplaceBeltramiProblem.cpp
  )
SET(HEADERS
  problems/HydrideProblem.hpp
  problems/HydMorphProblem.hpp
  problems/LaplaceBeltramiProblem.hpp
  )

IF(ALBANY_MPI)
  SET(SOURCES ${SOURCES}
    problems/MesoScaleLinkProblem.cpp
    )
  SET(HEADERS ${HEADERS}
    problems/MesoScaleLinkProblem.hpp
    )
ENDIF()

# evaluators
SET(SOURCES ${SOURCES}
  evaluators/HydrideCResid.cpp
  evaluators/HydrideWResid.cpp
  evaluators/HydrideChemTerm.cpp
  evaluators/HydrideStress.cpp
  evaluators/HydrideStressTerm.cpp
  evaluators/JThermConductivity.cpp
  evaluators/HydFractionResid.cpp
  evaluators/LaplaceResid.cpp
  evaluators/TPSLaplaceResid.cpp
  evaluators/TPSALaplaceResid.cpp
  evaluators/LaplaceBeltramiResid.cpp
  evaluators/ContravariantTargetMetricTensor.cpp
  evaluators/CalcInstantaneousCoords.cpp
  )

SET(HEADERS ${HEADERS}
  evaluators/HydrideCResid.hpp
  evaluators/HydrideCResid_Def.hpp
  evaluators/HydrideWResid.hpp
  evaluators/HydrideWResid_Def.hpp
  evaluators/HydrideChemTerm.hpp
  evaluators/HydrideChemTerm_Def.hpp
  evaluators/HydrideStress.hpp
  evaluators/HydrideStress_Def.hpp
  evaluators/HydrideStressTerm.hpp
  evaluators/HydrideStressTerm_Def.hpp
  evaluators/JThermConductivity.hpp
  evaluators/JThermConductivity_Def.hpp
  evaluators/HydFractionResid.hpp
  evaluators/HydFractionResid_Def.hpp
  evaluators/LaplaceResid.hpp
  evaluators/LaplaceResid_Def.hpp
  evaluators/TPSLaplaceResid.hpp
  evaluators/TPSLaplaceResid_Def.hpp
  evaluators/TPSALaplaceResid.hpp
  evaluators/TPSALaplaceResid_Def.hpp
  evaluators/LaplaceBeltramiResid.hpp
  evaluators/LaplaceBeltramiResid_Def.hpp
  evaluators/ContravariantTargetMetricTensor.hpp
  evaluators/ContravariantTargetMetricTensor_Def.hpp
  evaluators/CalcInstantaneousCoords.hpp
  evaluators/CalcInstantaneousCoords_Def.hpp
  )

IF(ALBANY_MPI)
  SET(SOURCES ${SOURCES}
    evaluators/MultiScaleStress.cpp
    )
  SET(HEADERS ${HEADERS}
    evaluators/MultiScaleStress.hpp
    evaluators/MultiScaleStress_Def.hpp
    )
ENDIF()


include_directories (${Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
  ${Albany_SOURCE_DIR}/src ${Albany_SOURCE_DIR}/src/evaluators
  ${Albany_SOURCE_DIR}/src/problems ${Albany_SOURCE_DIR}/src/responses
  ${Albany_SOURCE_DIR}/src/stk ${Albany_SOURCE_DIR}/src/LCM/
  ${Albany_SOURCE_DIR}/src/LCM/evaluators ${Albany_SOURCE_DIR}/src/LCM/utils
  ./problems ./evaluators
  )

add_library(hydride ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})

IF ( INSTALL_ALBANY )
    install(TARGETS hydride EXPORT Albany
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()
