
# Create a symbolic link for exodiff and epu
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SEACAS_EPU} ${CMAKE_CURRENT_BINARY_DIR}/epu)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBCPatchTestSetup.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/OBCPatchTestSetup.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBCPatchTestSetup_Materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/OBCPatchTestSetup_Materials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBCPatchTestSetup.g
               ${CMAKE_CURRENT_BINARY_DIR}/OBCPatchTestSetup.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBCPatchTestSetup.g.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/OBCPatchTestSetup.g.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBCPatchTestSetup.g.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/OBCPatchTestSetup.g.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBCPatchTestSetup.g.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/OBCPatchTestSetup.g.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBCPatchTestSetup.g.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/OBCPatchTestSetup.g.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBCPatchTestSetup.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/OBCPatchTestSetup.gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBCPatchTestSetup.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/OBCPatchTestSetup.exodiff COPYONLY)


# Add the tests
if(ALBANY_IFPACK2)
add_test(PeridigmCoupling_OBCPatchTestSetup_np1_Epetra python ../run_exodiff_test.py "${SerialAlbany.exe}" OBCPatchTestSetup.yaml)
add_test(PeridigmCoupling_OBCPatchTestSetup_np4_Epetra python ../run_exodiff_test.py "${Albany.exe}" OBCPatchTestSetup.yaml)
endif()
