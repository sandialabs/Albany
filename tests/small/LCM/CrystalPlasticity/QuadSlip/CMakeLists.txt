
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/QuadSlip.g
               ${CMAKE_CURRENT_BINARY_DIR}/QuadSlip.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/QuadSlip.xml
               ${CMAKE_CURRENT_BINARY_DIR}/QuadSlip.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/QuadSlip_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/QuadSlip_Material.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/QuadSlip.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/QuadSlip.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/QuadSlip.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/QuadSlip.gold.exo COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_QuadSlip python ../run_exodiff_test.py "${SerialAlbanyT.exe}" QuadSlip.xml)
endif()
