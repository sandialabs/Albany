
# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OptimizationBasedCoupling_PoissonMaterials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/OptimizationBasedCoupling_PoissonMaterials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OptimizationBasedCouplingCMWA_Analysis.xml
               ${CMAKE_CURRENT_BINARY_DIR}/OptimizationBasedCouplingCMWA_Analysis.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../CMWA/CMWA_1.g
               ${CMAKE_CURRENT_BINARY_DIR}/CMWA_1.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../CMWA/CMWA_1.g.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/CMWA_1.g.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../CMWA/CMWA_1.g.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/CMWA_1.g.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../CMWA/CMWA_1.g.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/CMWA_1.g.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../CMWA/CMWA_1.g.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/CMWA_1.g.4.3 COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
if (ALBANY_EPETRA) 
# add_test(${testName}_CMWA ${SerialAlbany.exe} OptimizationBasedCouplingCMWA_Analysis.xml)
add_test(${testName}_CMWA ${Albany.exe} OptimizationBasedCouplingCMWA_Analysis.xml)
endif()
