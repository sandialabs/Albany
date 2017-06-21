
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleHexElement.g
               ${CMAKE_CURRENT_BINARY_DIR}/SingleHexElement.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ElasticShear3d.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/ElasticShear3d.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CrystalElasticity3d_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/CrystalElasticity3d_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ElasticShear3d.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/ElasticShear3d.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ElasticShear3d.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/ElasticShear3d.gold.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AnalyticSolution.py
               ${CMAKE_CURRENT_BINARY_DIR}/AnalyticSolution.py COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ElasticShearRotated3d.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/ElasticShearRotated3d.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CrystalElasticityRotated3d_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/CrystalElasticityRotated3d_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ElasticShearRotated3d.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/ElasticShearRotated3d.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ElasticShearRotated3d.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/ElasticShearRotated3d.gold.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AnalyticSolution.py
               ${CMAKE_CURRENT_BINARY_DIR}/AnalyticSolution.py COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_ElasticShear3d python ../run_exodiff_test.py "${SerialAlbanyT.exe}" ElasticShear3d.yaml)
add_test(CrystalPlasticity_ElasticShearRotated3d python ../run_exodiff_test.py "${SerialAlbanyT.exe}" ElasticShearRotated3d.yaml)
endif()
