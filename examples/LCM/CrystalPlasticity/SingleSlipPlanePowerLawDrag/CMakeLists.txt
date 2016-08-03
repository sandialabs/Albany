
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlanePowerLawDrag.g
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlanePowerLawDrag.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlanePowerLawDrag_NumericalDrag.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlanePowerLawDrag_NumericalDrag.xml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlanePowerLawDrag_ActiveDrag.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlanePowerLawDrag_ActiveDrag.xml COPYONLY)
	     
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlanePowerLawDrag_NumericalDrag_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlanePowerLawDrag_NumericalDrag_Material.xml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlanePowerLawDrag_ActiveDrag_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlanePowerLawDrag_ActiveDrag_Material.xml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlanePowerLawDrag_NumericalDrag.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlanePowerLawDrag_NumericalDrag.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlanePowerLawDrag_ActiveDrag.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlanePowerLawDrag_ActiveDrag.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlanePowerLawDrag_NumericalDrag.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlanePowerLawDrag_NumericalDrag.exodiff COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlanePowerLawDrag_ActiveDrag.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlanePowerLawDrag_ActiveDrag.exodiff COPYONLY)
	     

# Add the tests
#if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_SingleSlipPlanePowerLawDrag_NumericalDrag python ../run_exodiff_test.py "${SerialAlbanyT.exe}" SingleSlipPlanePowerLawDrag_NumericalDrag.xml)
add_test(CrystalPlasticity_SingleSlipPlanePowerLawDrag_ActiveDrag python ../run_exodiff_test.py "${SerialAlbanyT.exe}" SingleSlipPlanePowerLawDrag_ActiveDrag.xml)
#endif()
