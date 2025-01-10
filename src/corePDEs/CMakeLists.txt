##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//


set(SOURCES
    problems/Albany_AdvDiffProblem.cpp
    problems/Albany_CoreProblemFactory.cpp
    problems/Albany_HeatProblem.cpp
    problems/Albany_PopulateMesh.cpp
    problems/Albany_SideLaplacianProblem.cpp
    evaluators/PHAL_AdvDiffResid.cpp
    evaluators/PHAL_HeatEqResid.cpp
    evaluators/PHAL_SideLaplacianResidual.cpp
    evaluators/PHAL_ThermalConductivity.cpp
  )

  set(HEADERS
    problems/Albany_AdvDiffProblem.hpp
    problems/Albany_CoreProblemFactory.hpp
    problems/Albany_HeatProblem.hpp
    problems/Albany_PopulateMesh.hpp
    problems/Albany_SideLaplacianProblem.hpp
    evaluators/PHAL_AdvDiffResid.hpp
    evaluators/PHAL_AdvDiffResid_Def.hpp
    evaluators/PHAL_HeatEqResid.hpp
    evaluators/PHAL_HeatEqResid_Def.hpp
    evaluators/PHAL_SideLaplacianResidual.hpp
    evaluators/PHAL_SideLaplacianResidual_Def.hpp
    evaluators/PHAL_ThermalConductivity.hpp
    evaluators/PHAL_ThermalConductivity_Def.hpp
  )

add_library(corePDEs ${Albany_LIBRARY_TYPE} ${SOURCES})
set_target_properties(corePDEs PROPERTIES PUBLIC_HEADER "${HEADERS}")
target_link_libraries(corePDEs PUBLIC albanyLib)
if (ALBANY_SUPPRESS_TRILINOS_WARNINGS)
	target_include_directories(corePDEs SYSTEM PUBLIC
                            "${Trilinos_INCLUDE_DIRS};${Trilinos_TPL_INCLUDE_DIRS}")
else()
	target_include_directories(corePDEs PUBLIC
                            "${Trilinos_INCLUDE_DIRS};${Trilinos_TPL_INCLUDE_DIRS}")
endif()

set (COREPDES_INCLUDE_DIRS
     ${ALBANY_INCLUDE_DIRS}
     ${CMAKE_CURRENT_SOURCE_DIR}/evaluators
     ${CMAKE_CURRENT_SOURCE_DIR}/problems
)

target_include_directories(corePDEs PUBLIC
	"$<BUILD_INTERFACE:${COREPDES_INCLUDE_DIRS}>"
                            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

set (ALBANY_LIBRARIES ${ALBANY_LIBRARIES} corePDEs PARENT_SCOPE)

install(TARGETS corePDEs EXPORT albany-export
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/"
        PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

