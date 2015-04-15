
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SEACAS_EPU} ${CMAKE_CURRENT_BINARY_DIR}/epu)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RubiksCube.xml
               ${CMAKE_CURRENT_BINARY_DIR}/RubiksCube.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RubiksCube_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/RubiksCube_Material.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RubiksCube.g
               ${CMAKE_CURRENT_BINARY_DIR}/RubiksCube.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RubiksCube.g.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/RubiksCube.g.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RubiksCube.g.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/RubiksCube.g.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RubiksCube.g.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/RubiksCube.g.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RubiksCube.g.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/RubiksCube.g.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RubiksCube.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/RubiksCube.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RubiksCube.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/RubiksCube.gold.exo COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_RubiksCube_np1_Tpetra python ../run_exodiff_test.py "${SerialAlbanyT.exe}" RubiksCube.xml)
#add_test(CrystalPlasticity_RubiksCube_np4_Tpetra python ../run_exodiff_test.py "${AlbanyT.exe}" RubiksCube.xml)
endif()
