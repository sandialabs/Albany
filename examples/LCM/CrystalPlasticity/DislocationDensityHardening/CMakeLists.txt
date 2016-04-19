
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/DislocationDensityHardening.g
               ${CMAKE_CURRENT_BINARY_DIR}/DislocationDensityHardening.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/DislocationDensityHardening.xml
               ${CMAKE_CURRENT_BINARY_DIR}/DislocationDensityHardening.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/DislocationDensityHardening_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/DislocationDensityHardening_Material.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/DislocationDensityHardening.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/DislocationDensityHardening.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/DislocationDensityHardening.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/DislocationDensityHardening.gold.exo COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_DislocationDensityHardening python ../run_exodiff_test.py "${SerialAlbanyT.exe}" DislocationDensityHardening.xml)
endif()
