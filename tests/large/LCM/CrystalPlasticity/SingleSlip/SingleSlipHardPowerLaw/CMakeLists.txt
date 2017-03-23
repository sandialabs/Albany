
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip.g
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHardPowerLaw_Implicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHardPowerLaw_Implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHardPowerLaw_Material_Implicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHardPowerLaw_Material_Implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHardPowerLaw_Implicit.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHardPowerLaw_Implicit.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHardPowerLaw_Implicit.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHardPowerLaw_Implicit.gold.exo COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_SingleSlipHardPowerLaw_Implicit_Tpetra python ../../run_exodiff_test.py "${SerialAlbanyT.exe}" SingleSlipHardPowerLaw_Implicit.yaml)
endif()
