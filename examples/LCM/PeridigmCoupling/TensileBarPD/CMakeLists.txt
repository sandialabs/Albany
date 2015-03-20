
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TensileBarPD.xml
               ${CMAKE_CURRENT_BINARY_DIR}/TensileBarPD.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TensileBarPDMaterials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/TensileBarPDMaterials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TensileBarPD.sphere.g
               ${CMAKE_CURRENT_BINARY_DIR}/TensileBarPD.sphere.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TensileBarPD.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/TensileBarPD.gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TensileBarPD_PeridigmResults.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/TensileBarPD_PeridigmResults.gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TensileBarPD.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/TensileBarPD.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TensileBarPD_PeridigmResults.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/TensileBarPD_PeridigmResults.exodiff COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(PeridigmCoupling_TensileBarPD_Epetra python ../run_exodiff_test.py "${SerialAlbany.exe}" TensileBarPD.xml)
endif()
