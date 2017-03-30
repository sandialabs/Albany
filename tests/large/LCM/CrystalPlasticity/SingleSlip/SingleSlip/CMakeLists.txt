
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip_Implicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip_Implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip_Implicit_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip_Implicit_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip_Implicit.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip_Implicit.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip_Implicit.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip_Implicit.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip_Explicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip_Explicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip_Explicit_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip_Explicit_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip_Explicit.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip_Explicit.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip_Explicit.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip_Explicit.gold.exo COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_SingleSlip_Explicit python ../../run_exodiff_test.py "${SerialAlbanyT.exe}" SingleSlip_Explicit.yaml)
add_test(CrystalPlasticity_SingleSlip_Implicit python ../../run_exodiff_test.py "${SerialAlbanyT.exe}" SingleSlip_Implicit.yaml)
endif()
