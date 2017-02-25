
# Create a symbolic link for exodiff and epu
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SEACAS_EPU} ${CMAKE_CURRENT_BINARY_DIR}/epu)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TensileBarPDPS.xml
               ${CMAKE_CURRENT_BINARY_DIR}/TensileBarPDPS.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TensileBarPDPSMaterials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/TensileBarPDPSMaterials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TensileBarPDPS.g
               ${CMAKE_CURRENT_BINARY_DIR}/TensileBarPDPS.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TensileBarPDPS.g.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/TensileBarPDPS.g.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TensileBarPDPS.g.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/TensileBarPDPS.g.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TensileBarPDPS.g.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/TensileBarPDPS.g.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TensileBarPDPS.g.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/TensileBarPDPS.g.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TensileBarPDPS.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/TensileBarPDPS.gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TensileBarPDPS.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/TensileBarPDPS.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TensileBarPDPS_PostProcess.py
               ${CMAKE_CURRENT_BINARY_DIR}/TensileBarPDPS_PostProcess.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TensileBarPDPS.gnuplot
               ${CMAKE_CURRENT_BINARY_DIR}/TensileBarPDPS.gnuplot COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/linear_solution.txt
               ${CMAKE_CURRENT_BINARY_DIR}/linear_solution.txt COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(PeridigmCoupling_TensileBarPDPS_np1_Epetra python ../run_exodiff_test.py "${SerialAlbany.exe}" TensileBarPDPS.xml)
#add_test(PeridigmCoupling_TensileBarPDPS_np4_Epetra python ../run_exodiff_test.py "${Albany.exe}" TensileBarPDPS.xml)
endif()
