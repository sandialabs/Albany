
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_exodiff_test.py
               ${CMAKE_CURRENT_BINARY_DIR}/run_exodiff_test.py COPYONLY)
set(runtest.cmake ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)

add_subdirectory(SingleElement2d)
add_subdirectory(SingleElement3d)
add_subdirectory(SingleSlip)
IF (NOT Kokkos_ENABLE_OpenMP) 
add_subdirectory(OrientationOnMesh)
add_subdirectory(SchwarzBar)
ENDIF()
add_subdirectory(RubiksCube)
add_subdirectory(MultiSlipHard)
add_subdirectory(MinisolverStep)
add_subdirectory(QuadSlipDislocationDensityTraction)
add_subdirectory(DislocationDensityHardening)
add_subdirectory(MultiFamily)
add_subdirectory(ThermallyActivatedSlip)
