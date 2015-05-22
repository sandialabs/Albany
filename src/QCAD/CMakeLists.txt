##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//
# QCAD

SET(SOURCES
  evaluators/QCAD_Permittivity.cpp
  evaluators/QCAD_PoissonResid.cpp
  evaluators/QCAD_PoissonSource.cpp
  evaluators/QCAD_PoissonDirichlet.cpp
  evaluators/QCAD_PoissonNeumann.cpp
  evaluators/QCAD_ResponseRegionBoundary.cpp
  evaluators/QCAD_SchrodingerPotential.cpp
  evaluators/QCAD_SchrodingerResid.cpp
  evaluators/QCAD_SchrodingerDirichlet.cpp
  evaluators/QCAD_StringFormulaEvaluator.cpp
  problems/QCAD_SchrodingerProblem.cpp
  problems/QCAD_PoissonProblem.cpp
  QCADT_ImplicitPSJacobian.cpp
  QCADT_CoupledPSJacobian.cpp
  QCADT_CoupledPoissonSchrodinger.cpp
  ../problems/Albany_ThermoElectrostaticsProblem.cpp
  ../evaluators/PHAL_JouleHeating.cpp
  ../evaluators/PHAL_TEProp.cpp
  )

IF (ALBANY_EPETRA)
  SET(SOURCES ${SOURCES} 
    QCAD_Solver.cpp
    QCADT_Solver.cpp
    QCAD_CoupledPoissonSchrodinger.cpp
    QCAD_CoupledPSJacobian.cpp
    QCAD_CoupledPSPreconditioner.cpp
    QCAD_CoupledPSObserver.cpp
    QCAD_MultiSolutionObserver.cpp
    QCAD_GenEigensolver.cpp
    evaluators/QCAD_ResponseSaddleValue.cpp
    responses/QCAD_SaddleValueResponseFunction.cpp
    responses/QCAD_GreensFunctionTunneling.cpp
  )
ENDIF()

SET(HEADERS
  QCADT_ImplicitPSJacobian.hpp
  QCADT_CoupledPSJacobian.hpp
  QCADT_CoupledPoissonSchrodinger.hpp
  evaluators/QCAD_Permittivity.hpp
  evaluators/QCAD_Permittivity_Def.hpp
  evaluators/QCAD_PoissonResid.hpp
  evaluators/QCAD_PoissonResid_Def.hpp
  evaluators/QCAD_PoissonSource.hpp
  evaluators/QCAD_PoissonSource_Def.hpp
  evaluators/QCAD_PoissonDirichlet.hpp
  evaluators/QCAD_PoissonDirichlet_Def.hpp
  evaluators/QCAD_PoissonNeumann.hpp
  evaluators/QCAD_PoissonNeumann_Def.hpp
  evaluators/QCAD_ResponseRegionBoundary.hpp
  evaluators/QCAD_ResponseRegionBoundary_Def.hpp
  evaluators/QCAD_SchrodingerPotential.hpp
  evaluators/QCAD_SchrodingerPotential_Def.hpp
  evaluators/QCAD_SchrodingerResid.hpp
  evaluators/QCAD_SchrodingerResid_Def.hpp
  evaluators/QCAD_SchrodingerDirichlet.hpp
  evaluators/QCAD_SchrodingerDirichlet_Def.hpp
  evaluators/QCAD_StringFormulaEvaluator.hpp
  problems/QCAD_SchrodingerProblem.hpp
  problems/QCAD_PoissonProblem.hpp
  ../problems/Albany_ThermoElectrostaticsProblem.hpp
  ../evaluators/PHAL_JouleHeating.hpp
  ../evaluators/PHAL_JouleHeating_Def.hpp
  ../evaluators/PHAL_TEProp.hpp
  ../evaluators/PHAL_TEProp_Def.hpp
  )

IF (ALBANY_EPETRA) 
  SET(HEADERS ${HEADERS} 
    QCAD_Solver.hpp
    QCADT_Solver.hpp
    QCAD_CoupledPoissonSchrodinger.hpp
    QCAD_CoupledPSJacobian.hpp
    QCAD_CoupledPSPreconditioner.hpp
    QCAD_CoupledPSObserver.hpp
    QCAD_MultiSolutionObserver.hpp
    QCAD_GenEigensolver.hpp
    evaluators/QCAD_ResponseSaddleValue.hpp
    evaluators/QCAD_ResponseSaddleValue_Def.hpp
    responses/QCAD_SaddleValueResponseFunction.hpp
    responses/QCAD_GreensFunctionTunneling.hpp
  )
ENDIF()

include_directories (${Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
  ${Albany_SOURCE_DIR}/src ${Albany_SOURCE_DIR}/src/evaluators 
  ${Albany_SOURCE_DIR}/src/problems ${Albany_SOURCE_DIR}/src/responses 
  ./problems ./evaluators ./responses
  )

add_library(QCAD ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})

set_target_properties(QCAD PROPERTIES PUBLIC_HEADER "${HEADERS}")

IF (INSTALL_ALBANY)
  install(TARGETS QCAD EXPORT albany-export
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()
