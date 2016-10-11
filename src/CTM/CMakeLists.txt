set(CTM_SOURCES
CTM_SolutionInfo.cpp
CTM_ThermalProblem.cpp
CTM_Application.cpp
CTM_Solver.cpp)

add_library(CTM ${CTM_SOURCES})
target_link_libraries(CTM PUBLIC ${ALBANY_LIBRARIES} ${ALL_LIBRARIES})

add_executable(CoupledThermoMechanicsSolver Main.cpp)
target_link_libraries(CoupledThermoMechanicsSolver CTM ${ALBANY_LIBRARIES} ${ALL_LIBRARIES})
