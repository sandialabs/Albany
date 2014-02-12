##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//
# FELIX

set(CISM_INCLUDE_DIRS /home/ikalash/Desktop/interface
    CISM_INCLUDE_DIRS /home/ikalash/Desktop/interface/FELIX
)
  SET(SOURCES 
       evaluators/FELIX_StokesMomentumResid.cpp
       evaluators/FELIX_StokesFOResid.cpp
       evaluators/FELIX_StokesL1L2Resid.cpp
       evaluators/FELIX_Viscosity.cpp
       evaluators/FELIX_ViscosityFO.cpp
       evaluators/FELIX_ViscosityL1L2.cpp
       evaluators/FELIX_EpsilonL1L2.cpp
       evaluators/FELIX_StokesContinuityResid.cpp
       evaluators/FELIX_StokesTauM.cpp
       evaluators/FELIX_StokesContravarientMetricTensor.cpp
       evaluators/FELIX_StokesRm.cpp
       evaluators/FELIX_StokesBodyForce.cpp
       evaluators/FELIX_StokesFOBodyForce.cpp
       evaluators/FELIX_StokesL1L2BodyForce.cpp
       problems/FELIX_Stokes.cpp
       problems/FELIX_StokesFO.cpp
       problems/FELIX_StokesL1L2.cpp
   )
   
   SET(SOURCESMPAS 
       interface_with_mpas/Interface.cpp
   )
IF(ENABLE_CISM_INTERFACE) 
   SET(SOURCESCISM 
       interface_with_cism/felix_driver.cpp
   )
ENDIF()

  SET(HEADERS
       evaluators/FELIX_StokesMomentumResid.hpp
       evaluators/FELIX_StokesFOResid_Def.hpp
       evaluators/FELIX_StokesFOResid.hpp
       evaluators/FELIX_StokesL1L2Resid_Def.hpp
       evaluators/FELIX_StokesL1L2Resid.hpp
       evaluators/FELIX_StokesMomentumResid_Def.hpp
       evaluators/FELIX_ViscosityFO.hpp
       evaluators/FELIX_ViscosityFO_Def.hpp
       evaluators/FELIX_ViscosityL1L2.hpp
       evaluators/FELIX_ViscosityL1L2_Def.hpp
       evaluators/FELIX_EpsilonL1L2.hpp
       evaluators/FELIX_EpsilonL1L2_Def.hpp
       evaluators/FELIX_StokesContinuityResid.hpp
       evaluators/FELIX_StokesContinuityResid_Def.hpp
       evaluators/FELIX_StokesTauM.hpp
       evaluators/FELIX_StokesTauM_Def.hpp
       evaluators/FELIX_StokesContravarientMetricTensor.hpp
       evaluators/FELIX_StokesContravarientMetricTensor_Def.hpp
       evaluators/FELIX_StokesRm.hpp
       evaluators/FELIX_StokesRm_Def.hpp
       evaluators/FELIX_StokesBodyForce.hpp
       evaluators/FELIX_StokesBodyForce_Def.hpp
       evaluators/FELIX_StokesFOBodyForce.hpp
       evaluators/FELIX_StokesFOBodyForce_Def.hpp
       problems/FELIX_Stokes.hpp
       problems/FELIX_StokesFO.hpp
       problems/FELIX_StokesL1L2.hpp
  )
  
  SET(HEADERSMPAS 
       interface_with_mpas/Interface.hpp
   )
  
IF (ENABLE_CISM_INTERFACE)
  SET(HEADERSCISM
       interface_with_cism/felix_driver.H
   )
ENDIF()

IF(ENABLE_CISM_INTERFACE)
  add_library(cismInterface ${Albany_LIBRARY_TYPE} ${SOURCESCISM} ${HEADERSCISM})
  set_target_properties(cismInterface PROPERTIES PUBLIC_HEADER "${HEADERSCISM}")
  include_directories (${Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
                       ${CISM_INCLUDE_DIR} ${CISM_INCLUDE_DIR}/FELIX
                       ${Albany_SOURCE_DIR}/src ${Albany_SOURCE_DIR}/src/evaluators 
                       ${Albany_SOURCE_DIR}/src/problems
                       ${Albany_SOURCE_DIR}/src/FELIX/interface_with_cism ${Albany_SOURCE_DIR}/src/responses 
                       ./problems ./evaluators
                      )
  IF (INSTALL_ALBANY)
    install(TARGETS cismInterface EXPORT albany-export
      LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
      ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
      PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
  ENDIF()
ELSE() 
  include_directories (${Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
                       ${Albany_SOURCE_DIR}/src ${Albany_SOURCE_DIR}/src/evaluators 
                       ${Albany_SOURCE_DIR}/src/problems ${Albany_SOURCE_DIR}/src/responses 
                       ./problems ./evaluators
                      )
ENDIF()

add_library(FELIX ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})
set_target_properties(FELIX PROPERTIES PUBLIC_HEADER "${HEADERS}")

add_library(mpasInterface ${Albany_LIBRARY_TYPE} ${SOURCESMPAS} ${HEADERSMPAS})
set_target_properties(mpasInterface PROPERTIES PUBLIC_HEADER "${HEADERSMPAS}")

IF (INSTALL_ALBANY)
  install(TARGETS FELIX mpasInterface EXPORT albany-export
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()


