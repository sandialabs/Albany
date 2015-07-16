
# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OptimizationBasedCoupling_PoissonMaterials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/OptimizationBasedCoupling_PoissonMaterials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OptimizationBasedCouplingCMWA_Analysis.xml
               ${CMAKE_CURRENT_BINARY_DIR}/OptimizationBasedCouplingCMWA_Analysis.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../CMWA/CMWA_1.g
               ${CMAKE_CURRENT_BINARY_DIR}/CMWA_1.g COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
if (ALBANY_EPETRA) 
add_test(${testName}_CMWA ${SerialAlbany.exe} OptimizationBasedCouplingCMWA_Analysis.xml)
set(PROCESSOR_COUNT 1)
endif()
