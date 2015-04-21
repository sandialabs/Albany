
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlaneOffKilterLoading.g
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlaneOffKilterLoading.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlaneOffKilterLoading.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlaneOffKilterLoading.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlaneOffKilterLoading_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlaneOffKilterLoading_Material.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlaneOffKilterLoading.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlaneOffKilterLoading.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlaneOffKilterLoading.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlaneOffKilterLoading.gold.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PostProcess.py
               ${CMAKE_CURRENT_BINARY_DIR}/PostProcess.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlaneOffKilterLoading.gnuplot
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlaneOffKilterLoading.gnuplot COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_SingleSlipPlaneOffKilterLoading_Tpetra python ../run_exodiff_test.py "${SerialAlbanyT.exe}" SingleSlipPlaneOffKilterLoading.xml)
endif()

