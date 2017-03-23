
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EPU} ${CMAKE_CURRENT_BINARY_DIR}/epu)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationOnMesh.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationOnMesh.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationOnMesh_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationOnMesh_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationOnMesh.g
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationOnMesh.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationOnMesh.g.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationOnMesh.g.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationOnMesh.g.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationOnMesh.g.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationOnMesh.g.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationOnMesh.g.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationOnMesh.g.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationOnMesh.g.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationOnMesh.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationOnMesh.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationOnMesh.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationOnMesh.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationNotOnMesh.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationNotOnMesh.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationNotOnMesh_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationNotOnMesh_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationNotOnMesh.g
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationNotOnMesh.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationNotOnMesh.g.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationNotOnMesh.g.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationNotOnMesh.g.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationNotOnMesh.g.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationNotOnMesh.g.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationNotOnMesh.g.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationNotOnMesh.g.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationNotOnMesh.g.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationNotOnMesh.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationNotOnMesh.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationNotOnMesh.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationNotOnMesh.gold.exo COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_OrientationOnMesh_Tpetra python ../run_exodiff_test.py "${SerialAlbanyT.exe}" OrientationOnMesh.yaml)
add_test(CrystalPlasticity_OrientationOnMesh_np4_Tpetra python ../run_exodiff_test.py "${AlbanyT.exe}" OrientationOnMesh.yaml)
add_test(CrystalPlasticity_OrientationNotOnMesh_Tpetra python ../run_exodiff_test.py "${SerialAlbanyT.exe}" OrientationNotOnMesh.yaml)
add_test(CrystalPlasticity_OrientationNotOnMesh_np4_Tpetra python ../run_exodiff_test.py "${AlbanyT.exe}" OrientationNotOnMesh.yaml)
endif()
