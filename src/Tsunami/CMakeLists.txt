##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//
# Tsunami


  SET(SOURCES
       problems/Tsunami_NavierStokes.cpp
       evaluators/Tsunami_NavierStokesContravarientMetricTensor.cpp
       evaluators/Tsunami_NavierStokesBodyForce.cpp
       evaluators/Tsunami_NavierStokesRm.cpp
       evaluators/Tsunami_NavierStokesContinuityResid.cpp
       evaluators/Tsunami_NavierStokesMomentumResid.cpp
       evaluators/Tsunami_NavierStokesTauPSPG.cpp
       evaluators/Tsunami_NavierStokesTauSUPG.cpp
   )

  SET(HEADERS
       problems/Tsunami_NavierStokes.hpp
       evaluators/Tsunami_NavierStokesContravarientMetricTensor.hpp
       evaluators/Tsunami_NavierStokesContravarientMetricTensor_Def.hpp
       evaluators/Tsunami_NavierStokesBodyForce.hpp
       evaluators/Tsunami_NavierStokesBodyForce_Def.hpp
       evaluators/Tsunami_NavierStokesRm.hpp
       evaluators/Tsunami_NavierStokesRm_Def.hpp
       evaluators/Tsunami_NavierStokesContinuityResid.hpp
       evaluators/Tsunami_NavierStokesContinuityResid_Def.hpp
       evaluators/Tsunami_NavierStokesMomentumResid.hpp
       evaluators/Tsunami_NavierStokesMomentumResid_Def.hpp
       evaluators/Tsunami_NavierStokesTauPSPG.hpp
       evaluators/Tsunami_NavierStokesTauPSPG_Def.hpp
       evaluators/Tsunami_NavierStokesTauSUPG.hpp
       evaluators/Tsunami_NavierStokesTauSUPG_Def.hpp
  )


include_directories (${Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
                     ${Albany_SOURCE_DIR}/src ${Albany_SOURCE_DIR}/src/evaluators 
                     ${Albany_SOURCE_DIR}/src/problems ${Albany_SOURCE_DIR}/src/responses 
                     ./problems ./evaluators
                    )

add_library(Tsunami ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})

set_target_properties(Tsunami PROPERTIES PUBLIC_HEADER "${HEADERS}")

IF (INSTALL_ALBANY)
  install(TARGETS Tsunami EXPORT albany-export
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()
