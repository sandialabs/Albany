
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip.g
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHard_Implicit.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHard_Implicit.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHard_Material_Implicit.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHard_Material_Implicit.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHard_Implicit.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHard_Implicit.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHard_Implicit.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHard_Implicit.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHard_Explicit.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHard_Explicit.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHard_Material_Explicit.xml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHard_Material_Explicit.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHard_Explicit.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHard_Explicit.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHard_Explicit.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHard_Explicit.gold.exo COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_SingleSlipHard_Explicit_Tpetra python ../../run_exodiff_test.py "${SerialAlbanyT.exe}" SingleSlipHard_Explicit.xml)
add_test(CrystalPlasticity_SingleSlipHard_Implicit_Tpetra python ../../run_exodiff_test.py "${SerialAlbanyT.exe}" SingleSlipHard_Implicit.xml)
endif()
