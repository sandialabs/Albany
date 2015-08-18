
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/blockDiffusion.exo
               ${CMAKE_CURRENT_BINARY_DIR}/blockDiffusion.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/blockDiffusionStab.xml
               ${CMAKE_CURRENT_BINARY_DIR}/blockDiffusionStab.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materialsScaledPlasticity.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materialsScaledPlasticity.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/blockDiffusionStab.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/blockDiffusionStab.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/blockDiffusionStab.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/blockDiffusionStab.gold.e COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(MechanicsWithHydrogenBar_requires_stabilization python ../run_exodiff_test.py "${SerialAlbanyT.exe}" blockDiffusionStab.xml)
endif()
