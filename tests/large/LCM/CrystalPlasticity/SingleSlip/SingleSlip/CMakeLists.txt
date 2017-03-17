
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip.g
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip_Implicit.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip_Implicit.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip_Implicit_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip_Implicit_Material.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip_Implicit.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip_Implicit.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip_Implicit.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip_Implicit.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip_Explicit.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip_Explicit.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip_Explicit_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip_Explicit_Material.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip_Explicit.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip_Explicit.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip_Explicit.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip_Explicit.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PostProcess.py
               ${CMAKE_CURRENT_BINARY_DIR}/PostProcess.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip.gnuplot
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip.gnuplot COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_SingleSlip_Explicit_Tpetra python ../../run_exodiff_test.py "${SerialAlbanyT.exe}" SingleSlip_Explicit.xml)
add_test(CrystalPlasticity_SingleSlip_Implicit_Tpetra python ../../run_exodiff_test.py "${SerialAlbanyT.exe}" SingleSlip_Implicit.xml)
endif()
