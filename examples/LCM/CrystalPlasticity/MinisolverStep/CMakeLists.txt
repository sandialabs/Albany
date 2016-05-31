
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_Specimen.g
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_Specimen.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_Newton.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_Newton.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_Newton_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_Newton_Material.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_Newton.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_Newton.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_Newton.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_Newton.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_TrustRegion.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_TrustRegion.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_TrustRegion_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_TrustRegion_Material.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_TrustRegion.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_TrustRegion.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_TrustRegion.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_TrustRegion.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_ConjugateGradient.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_ConjugateGradient.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_ConjugateGradient_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_ConjugateGradient_Material.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_ConjugateGradient.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_ConjugateGradient.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_ConjugateGradient.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_ConjugateGradient.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_LineSearchRegularized.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_LineSearchRegularized.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_LineSearchRegularized_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_LineSearchRegularized_Material.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_LineSearchRegularized.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_LineSearchRegularized.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MinisolverStep_LineSearchRegularized.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/MinisolverStep_LineSearchRegularized.gold.exo COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(CrystalPlasticity_MinisolverStep_Newton python ../run_exodiff_test.py "${SerialAlbanyT.exe}" MinisolverStep_Newton.xml)
add_test(CrystalPlasticity_MinisolverStep_TrustRegion python ../run_exodiff_test.py "${SerialAlbanyT.exe}" MinisolverStep_TrustRegion.xml)
#add_test(CrystalPlasticity_MinisolverStep_ConjugateGradient python ../run_exodiff_test.py "${SerialAlbanyT.exe}" MinisolverStep_ConjugateGradient.xml)
#add_test(CrystalPlasticity_MinisolverStep_LineSearchRegularized python ../run_exodiff_test.py "${SerialAlbanyT.exe}" MinisolverStep_LineSearchRegularized.xml)
endif()
