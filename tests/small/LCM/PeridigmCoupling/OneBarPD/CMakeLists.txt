
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OneBarPDMatrixFree.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/OneBarPDMatrixFree.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OneBarPDMaterials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/OneBarPDMaterials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OneBarPD.sphere.g
               ${CMAKE_CURRENT_BINARY_DIR}/OneBarPD.sphere.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OneBarPDMatrixFree.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/OneBarPDMatrixFree.gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OneBarPDMatrixFree.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/OneBarPDMatrixFree.exodiff COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(PeridigmCoupling_OneBarPD_Epetra python ../run_exodiff_test.py "${SerialAlbany.exe}" OneBarPDMatrixFree.yaml)
endif()
