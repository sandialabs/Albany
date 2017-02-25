
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ElasticShear2d.xml
               ${CMAKE_CURRENT_BINARY_DIR}/ElasticShear2d.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CrystalElasticity2d_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/CrystalElasticity2d_Material.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ElasticShear2d.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/ElasticShear2d.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ElasticShear2d.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/ElasticShear2d.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PlasticShear2d.xml
               ${CMAKE_CURRENT_BINARY_DIR}/PlasticShear2d.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CrystalPlasticity2d_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/CrystalPlasticity2d_Material.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PlasticShear2d.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/PlasticShear2d.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PlasticShear2d.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/PlasticShear2d.gold.exo COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_ElasticShear2d_Tpetra python ../run_exodiff_test.py "${SerialAlbanyT.exe}" ElasticShear2d.xml)
add_test(CrystalPlasticity_PlasticShear2d_Tpetra python ../run_exodiff_test.py "${SerialAlbanyT.exe}" PlasticShear2d.xml)
endif()
