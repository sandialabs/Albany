
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiFamily.g
               ${CMAKE_CURRENT_BINARY_DIR}/MultiFamily.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiFamily.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MultiFamily.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiFamily_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MultiFamily_Material.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiFamily.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/MultiFamily.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiFamily.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/MultiFamily.gold.exo COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_MultiFamily python ../run_exodiff_test.py "${SerialAlbanyT.exe}" MultiFamily.xml)
endif()
