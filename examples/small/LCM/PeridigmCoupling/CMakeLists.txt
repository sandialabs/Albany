
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_exodiff_test.py
               ${CMAKE_CURRENT_BINARY_DIR}/run_exodiff_test.py COPYONLY)

add_subdirectory(VectorPoisson)
add_subdirectory(OBCPatchTestSetup)
add_subdirectory(OBC_PatchTest)
add_subdirectory(OBC_Crack)
add_subdirectory(OBC_Mechanics_PatchTest)
add_subdirectory(OBC_Mechanics_Crack)
