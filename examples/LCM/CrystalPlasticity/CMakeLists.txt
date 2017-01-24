
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_exodiff_test.py
               ${CMAKE_CURRENT_BINARY_DIR}/run_exodiff_test.py COPYONLY)

add_subdirectory(SingleElement2d)
add_subdirectory(SingleElement3d)
add_subdirectory(SingleSlip)
add_subdirectory(OrientationOnMesh)
add_subdirectory(RubiksCube)
add_subdirectory(MultiSlipHard)
add_subdirectory(MinisolverStep)
add_subdirectory(QuadSlip)
add_subdirectory(DislocationDensityHardening)
add_subdirectory(MultiFamily)
add_subdirectory(ThermallyActivatedSlip)
add_subdirectory(SchwarzBar)
