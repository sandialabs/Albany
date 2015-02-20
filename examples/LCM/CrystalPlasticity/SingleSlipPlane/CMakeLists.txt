
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlane.g
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlane.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlane.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlane.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlane_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlane_Material.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlane.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlane.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlane.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlane.gold.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PostProcess.py
               ${CMAKE_CURRENT_BINARY_DIR}/PostProcess.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlane.gnuplot
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlane.gnuplot COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_SingleSlipPlane_Tpetra python ../run_exodiff_test.py "${SerialAlbanyT.exe}" SingleSlipPlane.xml)
endif()
