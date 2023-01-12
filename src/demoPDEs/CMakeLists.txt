##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//


set(SOURCES
    problems/Albany_DemoProblemFactory.cpp
    problems/Albany_AdvDiffProblem.cpp
    problems/Albany_ReactDiffSystem.cpp
    problems/Albany_Helmholtz2DProblem.cpp
    problems/Albany_ODEProblem.cpp
    problems/Albany_NavierStokes.cpp
    problems/Albany_ThermoElectrostaticsProblem.cpp
    problems/Albany_ThermalProblem.cpp
    problems/Albany_AdvectionProblem.cpp
    evaluators/PHAL_AdvDiffResid.cpp
    evaluators/PHAL_ReactDiffSystemResid.cpp
    evaluators/PHAL_HelmholtzResid.cpp
    evaluators/PHAL_JouleHeating.cpp
    evaluators/PHAL_NSBodyForce.cpp
    evaluators/PHAL_NSContinuityResid.cpp
    evaluators/PHAL_NSContravarientMetricTensor.cpp
    evaluators/PHAL_NSForchheimerTerm.cpp
    evaluators/PHAL_NSMaterialProperty.cpp
    evaluators/PHAL_NSMomentumResid.cpp
    evaluators/PHAL_NSNeutronEqResid.cpp
    evaluators/PHAL_NSPermeabilityTerm.cpp
    evaluators/PHAL_ThermalResid.cpp
    evaluators/PHAL_AdvectionResid.cpp
    evaluators/PHAL_NSRm.cpp
    evaluators/PHAL_NSTauM.cpp
    evaluators/PHAL_NSTauT.cpp
    evaluators/PHAL_NSThermalEqResid.cpp
    evaluators/PHAL_ODEResid.cpp
    evaluators/PHAL_PoissonResid.cpp
    evaluators/PHAL_TEProp.cpp
  )

  set(HEADERS
    problems/Albany_DemoProblemFactory.hpp
    problems/Albany_AdvDiffProblem.hpp
    problems/Albany_ReactDiffSystem.hpp
    problems/Albany_Helmholtz2DProblem.hpp
    problems/Albany_NavierStokes.hpp
    problems/Albany_ODEProblem.hpp
    problems/Albany_ThermoElectrostaticsProblem.hpp
    problems/Albany_ThermalProblem.hpp
    problems/Albany_AdvectionProblem.hpp
    evaluators/PHAL_AdvDiffResid.hpp
    evaluators/PHAL_AdvDiffResid_Def.hpp
    evaluators/PHAL_ReactDiffSystemResid.hpp
    evaluators/PHAL_ReactDiffSystemResid_Def.hpp
    evaluators/PHAL_HelmholtzResid.hpp
    evaluators/PHAL_HelmholtzResid_Def.hpp
    evaluators/PHAL_JouleHeating.hpp
    evaluators/PHAL_JouleHeating_Def.hpp
    evaluators/PHAL_NSContinuityResid.hpp
    evaluators/PHAL_NSContinuityResid_Def.hpp
    evaluators/PHAL_NSBodyForce.hpp
    evaluators/PHAL_NSBodyForce_Def.hpp
    evaluators/PHAL_NSContravarientMetricTensor.hpp
    evaluators/PHAL_NSContravarientMetricTensor_Def.hpp
    evaluators/PHAL_NSForchheimerTerm.hpp
    evaluators/PHAL_NSForchheimerTerm_Def.hpp
    evaluators/PHAL_NSMaterialProperty.hpp
    evaluators/PHAL_NSMaterialProperty_Def.hpp
    evaluators/PHAL_NSMomentumResid.hpp
    evaluators/PHAL_NSMomentumResid_Def.hpp
    evaluators/PHAL_NSNeutronEqResid.hpp
    evaluators/PHAL_NSNeutronEqResid_Def.hpp
    evaluators/PHAL_NSPermeabilityTerm.hpp
    evaluators/PHAL_NSPermeabilityTerm_Def.hpp
    evaluators/PHAL_NSRm.hpp
    evaluators/PHAL_NSRm_Def.hpp
    evaluators/PHAL_NSTauM.hpp
    evaluators/PHAL_NSTauM_Def.hpp
    evaluators/PHAL_NSTauT.hpp
    evaluators/PHAL_NSTauT_Def.hpp
    evaluators/PHAL_NSThermalEqResid.hpp
    evaluators/PHAL_NSThermalEqResid_Def.hpp
    evaluators/PHAL_ODEResid.hpp
    evaluators/PHAL_ODEResid_Def.hpp
    evaluators/PHAL_PoissonResid.hpp
    evaluators/PHAL_PoissonResid_Def.hpp
    evaluators/PHAL_TEProp.hpp
    evaluators/PHAL_TEProp_Def.hpp
    evaluators/PHAL_ThermalResid.hpp
    evaluators/PHAL_ThermalResid_Def.hpp
    evaluators/PHAL_AdvectionResid.hpp
    evaluators/PHAL_AdvectionResid_Def.hpp
  )

add_library(demoPDEs ${Albany_LIBRARY_TYPE} ${SOURCES})
set_target_properties(demoPDEs PROPERTIES PUBLIC_HEADER "${HEADERS}")
target_link_libraries(demoPDEs PUBLIC albanyLib)
if (ALBANY_SUPPRESS_TRILINOS_WARNINGS)
	target_include_directories(demoPDEs SYSTEM PUBLIC
                            "${Trilinos_INCLUDE_DIRS};${Trilinos_TPL_INCLUDE_DIRS}")
else()
	target_include_directories(demoPDEs PUBLIC
                            "${Trilinos_INCLUDE_DIRS};${Trilinos_TPL_INCLUDE_DIRS}")
endif()
target_compile_definitions (demoPDEs PUBLIC  ALBANY_DEMO_PDES)

set (DEMOPDES_INCLUDE_DIRS
     ${ALBANY_INCLUDE_DIRS}
     ${Albany_SOURCE_DIR}/src/corePDEs/evaluators
     ${CMAKE_CURRENT_SOURCE_DIR}/evaluators
     ${CMAKE_CURRENT_SOURCE_DIR}/problems
)

target_include_directories(demoPDEs PUBLIC
	"$<BUILD_INTERFACE:${DEMOPDES_INCLUDE_DIRS}>"
                            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

set (ALBANY_LIBRARIES ${ALBANY_LIBRARIES} demoPDEs PARENT_SCOPE)

install(TARGETS demoPDEs EXPORT albany-export
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/"
        PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

