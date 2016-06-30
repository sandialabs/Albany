
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationOnMesh.xml
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationOnMesh.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationOnMesh_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationOnMesh_Material.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationOnMesh.g
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationOnMesh.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationOnMesh.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationOnMesh.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationOnMesh.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationOnMesh.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationNotOnMesh.xml
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationNotOnMesh.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationNotOnMesh_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationNotOnMesh_Material.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationNotOnMesh.g
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationNotOnMesh.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationNotOnMesh.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationNotOnMesh.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationNotOnMesh.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationNotOnMesh.gold.exo COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_OrientationOnMesh_Tpetra python ../run_exodiff_test.py "${SerialAlbanyT.exe}" OrientationOnMesh.xml)
add_test(CrystalPlasticity_OrientationNotOnMesh_Tpetra python ../run_exodiff_test.py "${SerialAlbanyT.exe}" OrientationNotOnMesh.xml)
endif()
