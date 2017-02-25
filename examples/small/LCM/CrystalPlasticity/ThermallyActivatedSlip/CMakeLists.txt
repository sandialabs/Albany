
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ThermallyActivatedSlip.g
               ${CMAKE_CURRENT_BINARY_DIR}/ThermallyActivatedSlip.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ThermallyActivatedSlip.xml
               ${CMAKE_CURRENT_BINARY_DIR}/ThermallyActivatedSlip.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ThermallyActivatedSlip_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/ThermallyActivatedSlip_Material.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ThermallyActivatedSlip.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/ThermallyActivatedSlip.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ThermallyActivatedSlip.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/ThermallyActivatedSlip.gold.exo COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_ThermallyActivatedSlip python ../run_exodiff_test.py "${SerialAlbanyT.exe}" ThermallyActivatedSlip.xml)
endif()
