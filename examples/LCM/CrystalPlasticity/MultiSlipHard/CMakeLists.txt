
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiSlipPlaneHard.g
               ${CMAKE_CURRENT_BINARY_DIR}/MultiSlipPlaneHard.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiSlipPlaneHardRev.g
               ${CMAKE_CURRENT_BINARY_DIR}/MultiSlipPlaneHardRev.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiSlipPlaneHard_Implicit.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MultiSlipPlaneHard_Implicit.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiSlipPlaneHard_Material_Implicit.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MultiSlipPlaneHard_Material_Implicit.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiSlipPlaneHard_Implicit.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/MultiSlipPlaneHard_Implicit.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiSlipPlaneHard_Implicit.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/MultiSlipPlaneHard_Implicit.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiSlipPlaneHard_Implicit_Active_Sets.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MultiSlipPlaneHard_Implicit_Active_Sets.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiSlipPlaneHard_Material_Implicit_Active_Sets.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MultiSlipPlaneHard_Material_Implicit_Active_Sets.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiSlipPlaneHard_Implicit_Active_Sets.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/MultiSlipPlaneHard_Implicit_Active_Sets.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiSlipPlaneHard_Implicit_Active_Sets.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/MultiSlipPlaneHard_Implicit_Active_Sets.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/plot_hard_implicit.py
               ${CMAKE_CURRENT_BINARY_DIR}/plot_hard_implicit.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/examineCondition.py
               ${CMAKE_CURRENT_BINARY_DIR}/examineCondition.py COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_MultiSlipPlaneHard_Active_Sets_Tpetra python ../run_exodiff_test.py "${SerialAlbanyT.exe}" MultiSlipPlaneHard_Implicit_Active_Sets.xml)
add_test(CrystalPlasticity_MultiSlipPlaneHard_Implicit_Tpetra python ../run_exodiff_test.py "${SerialAlbanyT.exe}" MultiSlipPlaneHard_Implicit.xml)
endif()
