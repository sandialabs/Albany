
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip.g
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipSaturation.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipSaturation.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipSaturation_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipSaturation_Material.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipSaturation.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipSaturation.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipSaturation.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipSaturation.gold.exo COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_SingleSlipSaturation python ../../run_exodiff_test.py "${SerialAlbanyT.exe}" SingleSlipSaturation.xml)
endif()
