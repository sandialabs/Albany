
# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OptimizationBasedCoupling_PoissonMaterials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/OptimizationBasedCoupling_PoissonMaterials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OptimizationBasedCouplingCMWA_Analysis.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/OptimizationBasedCouplingCMWA_Analysis.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CMWA_1.g
               ${CMAKE_CURRENT_BINARY_DIR}/CMWA_1.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OptimizationBasedCouplingCMWA_Analysis.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/OptimizationBasedCouplingCMWA_Analysis.gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OptimizationBasedCouplingCMWA_Analysis.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/OptimizationBasedCouplingCMWA_Analysis.exodiff COPYONLY)


get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
if (ALBANY_EPETRA) 
add_test(PeridigmCoupling_OptimizationBasedCoupling_np1_Epetra python ../run_exodiff_test.py "${SerialAlbany.exe}" OptimizationBasedCouplingCMWA_Analysis.yaml)
add_test(PeridigmCoupling_OptimizationBasedCoupling_np4_Epetra python ../run_exodiff_test.py "${Albany.exe}" OptimizationBasedCouplingCMWA_Analysis.yaml)
endif()
