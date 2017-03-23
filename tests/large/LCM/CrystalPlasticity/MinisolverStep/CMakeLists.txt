
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_Specimen.g
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_Specimen.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_Newton.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_Newton.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_Newton_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_Newton_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_Newton.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_Newton.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_Newton.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_Newton.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_NewtonLineSearch.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_NewtonLineSearch.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_NewtonLineSearch_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_NewtonLineSearch_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_NewtonLineSearch.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_NewtonLineSearch.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_NewtonLineSearch.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_NewtonLineSearch.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_TrustRegion.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_TrustRegion.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_TrustRegion_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_TrustRegion_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_TrustRegion.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_TrustRegion.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_TrustRegion.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_TrustRegion.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_ConjugateGradient.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_ConjugateGradient.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_ConjugateGradient_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_ConjugateGradient_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_ConjugateGradient.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_ConjugateGradient.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_ConjugateGradient.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_ConjugateGradient.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_LineSearchRegularized.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_LineSearchRegularized.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_LineSearchRegularized_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_LineSearchRegularized_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_LineSearchRegularized.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_LineSearchRegularized.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_LineSearchRegularized.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_LineSearchRegularized.gold.exo COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_MinisolverStep_Newton python ../run_exodiff_test.py "${SerialAlbanyT.exe}" MinisolverStep_Newton.yaml)
add_test(CrystalPlasticity_MinisolverStep_NewtonLineSearch python ../run_exodiff_test.py "${SerialAlbanyT.exe}" MinisolverStep_NewtonLineSearch.yaml)
add_test(CrystalPlasticity_MinisolverStep_TrustRegion python ../run_exodiff_test.py "${SerialAlbanyT.exe}" MinisolverStep_TrustRegion.yaml)
#add_test(CrystalPlasticity_MinisolverStep_ConjugateGradient python ../run_exodiff_test.py "${SerialAlbanyT.exe}" MinisolverStep_ConjugateGradient.yaml)
#add_test(CrystalPlasticity_MinisolverStep_LineSearchRegularized python ../run_exodiff_test.py "${SerialAlbanyT.exe}" MinisolverStep_LineSearchRegularized.yaml)
endif()
