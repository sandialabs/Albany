
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipOffKilterLoading.g
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipOffKilterLoading.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipOffKilterLoading.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipOffKilterLoading.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipOffKilterLoading_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipOffKilterLoading_Material.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipOffKilterLoading.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipOffKilterLoading.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipOffKilterLoading.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipOffKilterLoading.gold.exo COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_SingleSlipOffKilterLoading_Tpetra python ../../run_exodiff_test.py "${SerialAlbanyT.exe}" SingleSlipOffKilterLoading.xml)
endif()

