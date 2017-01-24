
# Create a symbolic link for exodiff, epu, and alegbra
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SEACAS_EPU} ${CMAKE_CURRENT_BINARY_DIR}/epu)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SEACAS_ALGEBRA} ${CMAKE_CURRENT_BINARY_DIR}/algebra)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBC_Mechanics_Crack_Analysis.xml
               ${CMAKE_CURRENT_BINARY_DIR}/OBC_Mechanics_Crack_Analysis.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBC_Mechanics_Crack_ElasticMaterials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/OBC_Mechanics_Crack_ElasticMaterials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBC_Mechanics_Crack.g
               ${CMAKE_CURRENT_BINARY_DIR}/OBC_Mechanics_Crack.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBC_Mechanics_Crack.g.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/OBC_Mechanics_Crack.g.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBC_Mechanics_Crack.g.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/OBC_Mechanics_Crack.g.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBC_Mechanics_Crack.g.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/OBC_Mechanics_Crack.g.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBC_Mechanics_Crack.g.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/OBC_Mechanics_Crack.g.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBC_Mechanics_Crack_Analysis.final_step.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/OBC_Mechanics_Crack_Analysis.final_step.gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OBC_Mechanics_Crack_Analysis.final_step.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/OBC_Mechanics_Crack_Analysis.final_step.exodiff COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(PeridigmCoupling_OBC_Mechanics_Crack_np1_Epetra python ../run_exodiff_test.py "${SerialAlbanyAnalysis.exe}" OBC_Mechanics_Crack_Analysis.xml --final_step_only)
add_test(PeridigmCoupling_OBC_Mechanics_Crack_np4_Epetra python ../run_exodiff_test.py "${AlbanyAnalysis.exe}" OBC_Mechanics_Crack_Analysis.xml --final_step_only)
endif()
