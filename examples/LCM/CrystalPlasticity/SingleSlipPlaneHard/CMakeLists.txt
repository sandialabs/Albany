
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlane.g
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlane.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlaneHard_Implicit.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlaneHard_Implicit.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlaneHard_Material_Implicit.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlaneHard_Material_Implicit.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlaneHard_Implicit.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlaneHard_Implicit.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlaneHard_Implicit.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlaneHard_Implicit.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlaneHard_Explicit.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlaneHard_Explicit.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlaneHard_Material_Explicit.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlaneHard_Material_Explicit.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlaneHard_Explicit.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlaneHard_Explicit.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipPlaneHard_Explicit.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipPlaneHard_Explicit.gold.exo COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_SingleSlipPlaneHard_Explicit_Tpetra python ../run_exodiff_test.py "${SerialAlbanyT.exe}" SingleSlipPlaneHard_Explicit.xml)
add_test(CrystalPlasticity_SingleSlipPlaneHard_Implicit_Tpetra python ../run_exodiff_test.py "${SerialAlbanyT.exe}" SingleSlipPlaneHard_Implicit.xml)
endif()
