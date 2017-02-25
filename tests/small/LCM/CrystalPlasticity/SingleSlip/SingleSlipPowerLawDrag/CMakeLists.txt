
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPowerLawDrag.g
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPowerLawDrag.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPowerLawDrag_NumericalDrag.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPowerLawDrag_NumericalDrag.xml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPowerLawDrag_ActiveDrag.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPowerLawDrag_ActiveDrag.xml COPYONLY)
	     
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPowerLawDrag_NumericalDrag_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPowerLawDrag_NumericalDrag_Material.xml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPowerLawDrag_ActiveDrag_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPowerLawDrag_ActiveDrag_Material.xml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPowerLawDrag_NumericalDrag.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPowerLawDrag_NumericalDrag.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPowerLawDrag_ActiveDrag.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPowerLawDrag_ActiveDrag.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPowerLawDrag_NumericalDrag.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPowerLawDrag_NumericalDrag.exodiff COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPowerLawDrag_ActiveDrag.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPowerLawDrag_ActiveDrag.exodiff COPYONLY)
	     

# Add the tests
#if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_SingleSlipPowerLawDrag_NumericalDrag python ../../run_exodiff_test.py "${SerialAlbanyT.exe}" SingleSlipPowerLawDrag_NumericalDrag.xml)
add_test(CrystalPlasticity_SingleSlipPowerLawDrag_ActiveDrag python ../../run_exodiff_test.py "${SerialAlbanyT.exe}" SingleSlipPowerLawDrag_ActiveDrag.xml)
#endif()
