
# Create a symbolic link for exodiff and epu
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SEACAS_EPU} ${CMAKE_CURRENT_BINARY_DIR}/epu)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/VectorPoisson.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/VectorPoisson.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/VectorPoissonMaterials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/VectorPoissonMaterials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/VectorPoisson.g
               ${CMAKE_CURRENT_BINARY_DIR}/VectorPoisson.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/VectorPoisson.g.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/VectorPoisson.g.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/VectorPoisson.g.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/VectorPoisson.g.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/VectorPoisson.g.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/VectorPoisson.g.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/VectorPoisson.g.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/VectorPoisson.g.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/VectorPoisson.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/VectorPoisson.gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/VectorPoisson.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/VectorPoisson.exodiff COPYONLY)


# Add the tests
if(ALBANY_IFPACK2)
add_test(PeridigmCoupling_VectorPoisson_np1_Epetra python ../run_exodiff_test.py "${SerialAlbany.exe}" VectorPoisson.yaml)
add_test(PeridigmCoupling_VectorPoisson_np4_Epetra python ../run_exodiff_test.py "${Albany.exe}" VectorPoisson.yaml)
endif()
