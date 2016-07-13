
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlanePowerLawDrag.g
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlanePowerLawDrag.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlanePowerLawDrag_Implicit.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlanePowerLawDrag_Implicit.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlanePowerLawDrag_Implicit_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlanePowerLawDrag_Implicit_Material.xml COPYONLY)

# Add the tests
#if(ALBANY_IFPACK2)
#add_test(CrystalPlasticity_SingleSlipPlane_Implicit_Tpetra python ../run_exodiff_test.py "${SerialAlbanyT.exe}" SingleSlipPlanePowerLawDrag_Implicit.xml)
#endif()
