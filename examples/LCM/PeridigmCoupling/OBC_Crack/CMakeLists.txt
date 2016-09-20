
# Create a symbolic link for exodiff and epu
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SEACAS_EPU} ${CMAKE_CURRENT_BINARY_DIR}/epu)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBC_Crack_Analysis.xml
               ${CMAKE_CURRENT_BINARY_DIR}/OBC_Crack_Analysis.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBC_Crack_PoissonMaterials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/OBC_Crack_PoissonMaterials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBC_Crack.g
               ${CMAKE_CURRENT_BINARY_DIR}/OBC_Crack.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBC_Crack.g.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/OBC_Crack.g.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBC_Crack.g.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/OBC_Crack.g.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBC_Crack.g.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/OBC_Crack.g.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBC_Crack.g.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/OBC_Crack.g.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBC_Crack_Analysis.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/OBC_Crack_Analysis.gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBC_Crack_PostProcess.py
               ${CMAKE_CURRENT_BINARY_DIR}/OBC_Crack_PostProcess.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBC_Crack.gnuplot
               ${CMAKE_CURRENT_BINARY_DIR}/OBC_Crack.gnuplot COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBC_Crack_Analysis.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/OBC_Crack_Analysis.exodiff COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
#add_test(PeridigmCoupling_OBC_Crack_np1_Epetra python ../run_exodiff_test.py "${SerialAlbanyAnalysis.exe}" OBC_Crack_Analysis.xml)
add_test(PeridigmCoupling_OBC_Crack_np4_Epetra python ../run_exodiff_test.py "${AlbanyAnalysis.exe}" OBC_Crack_Analysis.xml)
endif()
