
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_exodiff_test.py
               ${CMAKE_CURRENT_BINARY_DIR}/run_exodiff_test.py COPYONLY)

add_subdirectory(VectorPoisson)
add_subdirectory(OBCPatchTestSetup)
add_subdirectory(OBC_PatchTest)
add_subdirectory(OBC_Crack)
add_subdirectory(OBC_Mechanics_PatchTest)
add_subdirectory(OBC_Mechanics_Crack)

#add_subdirectory(TensileBarPD)
#add_subdirectory(TensileBarPS)
#add_subdirectory(TensileBarPDPS)

# These tests no longer function as of Jan 2016, presumably due to changes in explicit time integration
#add_subdirectory(WaveInBarFEM)
#add_subdirectory(WaveInBarPS)
#add_subdirectory(WaveInBarPD)
#add_subdirectory(WaveInBarFEMPS)
#add_subdirectory(WaveInBarFEMPSPD)
