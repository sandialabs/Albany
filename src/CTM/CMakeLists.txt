set(CTM_SOURCES
CTM_Solver.cpp)

add_library(CTM ${CTM_SOURCES})
target_link_libraries(CTM PUBLIC ${ALBANY_LIBRARIES} ${ALL_LIBRARIES})

add_executable(CoupledThermoMechanicsSolver CoupledThermoMechanicsSolver.cpp)
target_link_libraries(CoupledThermoMechanicsSolver CTM ${ALBANY_LIBRARIES} ${ALL_LIBRARIES})
