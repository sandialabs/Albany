
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlane.g
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlane.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlaneHardPowerLaw_Implicit.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlaneHardPowerLaw_Implicit.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlaneHardPowerLaw_Material_Implicit.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlaneHardPowerLaw_Material_Implicit.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlaneHardPowerLaw_Implicit.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlaneHardPowerLaw_Implicit.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlaneHardPowerLaw_Implicit.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlaneHardPowerLaw_Implicit.gold.exo COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_SingleSlipPlaneHardPowerLaw_Implicit_Tpetra python ../run_exodiff_test.py "${SerialAlbanyT.exe}" SingleSlipPlaneHardPowerLaw_Implicit.xml)
endif()
