set(CTM_SOURCES
CTM_SolutionInfo.cpp
CTM_TemperatureEvaluator.cpp
CTM_ThermalProblem.cpp
CTM_MechanicsProblem.cpp
CTM_LinearSolver.cpp
CTM_Assembler.cpp
CTM_Adapter.cpp
CTM_Solver.cpp)

include_directories(../LCM/problems)
add_library(CTM ${CTM_SOURCES})
target_link_libraries(CTM PUBLIC ${ALBANY_LIBRARIES} ${ALL_LIBRARIES})

add_executable(CTMSolve Main.cpp)
target_link_libraries(CTMSolve CTM ${ALBANY_LIBRARIES} ${ALL_LIBRARIES})
