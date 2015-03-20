
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OneBarPS.xml
               ${CMAKE_CURRENT_BINARY_DIR}/OneBarPS.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OneBarPSMaterials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/OneBarPSMaterials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OneBarPS.g
               ${CMAKE_CURRENT_BINARY_DIR}/OneBarPS.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OneBarPS.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/OneBarPS.gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OneBarPS_PeridigmResults.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/OneBarPS_PeridigmResults.gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OneBarPS.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/OneBarPS.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OneBarPS_PeridigmResults.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/OneBarPS_PeridigmResults.exodiff COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(PeridigmCoupling_OneBarPS_Epetra python ../run_exodiff_test.py "${SerialAlbany.exe}" OneBarPS.xml)
endif()
