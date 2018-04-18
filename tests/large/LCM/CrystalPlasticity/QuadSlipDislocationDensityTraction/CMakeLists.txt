
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SEACAS_EPU} ${CMAKE_CURRENT_BINARY_DIR}/epu)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/QuadSlipDislocationDensityTraction.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/QuadSlipDislocationDensityTraction.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/QuadSlipDislocationDensity_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/QuadSlipDislocationDensity_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/QuadSlip.g
               ${CMAKE_CURRENT_BINARY_DIR}/QuadSlip.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/QuadSlipDislocationDensityTraction.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/QuadSlipDislocationDensityTraction.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/QuadSlipDislocationDensityTraction.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/QuadSlipDislocationDensityTraction.gold.exo COPYONLY)	       

# Add the tests
if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_QuadSlipDislocationDensityTraction python ../run_exodiff_test.py "${SerialAlbanyT.exe}" QuadSlipDislocationDensityTraction.yaml)
endif()
