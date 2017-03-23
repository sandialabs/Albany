
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/blockDiffusion.exo
               ${CMAKE_CURRENT_BINARY_DIR}/blockDiffusion.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/blockDiffusionNoStab.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/blockDiffusionNoStab.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materialsScaledPlasticity.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materialsScaledPlasticity.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/blockDiffusionNoStab.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/blockDiffusionNoStab.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/blockDiffusionNoStab.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/blockDiffusionNoStab.gold.e COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(MechanicsWithHydrogenBar_no_stabilization python ../run_exodiff_test.py "${SerialAlbanyT.exe}" blockDiffusionNoStab.yaml)
endif()
