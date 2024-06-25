# The root name for the tests in this directory is the directory name
get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${parentDirName}_${dirName})

#
# Tempus tests
#

# BE test
set(testName ${testNameRoot}_Tempus_BackwardEuler_NOXSolver)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tempus_be_nox_solver.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/tempus_be_nox_solver.yaml COPYONLY)

add_test(${testName} ${Albany.exe} tempus_be_nox_solver.yaml)
set_tests_properties(${testName} PROPERTIES LABELS
                                            "Basic;Tempus;Forward")


if (ALBANY_ROL)
    set(testName ${testNameRoot}_Transient_Analysis_Integrated_Response_Scalar_Params)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tempus_be_nox_analysis_integrated_response_scalar_params.yaml
                ${CMAKE_CURRENT_BINARY_DIR}/tempus_be_nox_analysis_integrated_response_scalar_params.yaml COPYONLY)

    add_test(${testName} ${AlbanyAnalysis.exe} tempus_be_nox_analysis_integrated_response_scalar_params.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "Transient;Analysis;Integrated;Scalar")


    set(testName ${testNameRoot}_Transient_Analysis_Integrated_Response_Distributed_Param)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tempus_be_nox_analysis_integrated_response_distributed_param.yaml
                ${CMAKE_CURRENT_BINARY_DIR}/tempus_be_nox_analysis_integrated_response_distributed_param.yaml COPYONLY)

    add_test(${testName} ${AlbanyAnalysis.exe} tempus_be_nox_analysis_integrated_response_distributed_param.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "Transient;Analysis;Integrated;Distributed")


    set(testName ${testNameRoot}_Transient_Analysis_ROL_Scalar_Params)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tempus_be_nox_analysis_rol_driven_scalar_params.yaml
                ${CMAKE_CURRENT_BINARY_DIR}/tempus_be_nox_analysis_rol_driven_scalar_params.yaml COPYONLY)

    add_test(${testName} ${AlbanyAnalysis.exe} tempus_be_nox_analysis_rol_driven_scalar_params.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "Transient;Analysis;ROL;Scalar")


    set(testName ${testNameRoot}_Transient_Analysis_Tempus_Scalar_Params)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tempus_be_nox_analysis_tempus_driven_scalar_params.yaml
                ${CMAKE_CURRENT_BINARY_DIR}/tempus_be_nox_analysis_tempus_driven_scalar_params.yaml COPYONLY)

    add_test(${testName} ${AlbanyAnalysis.exe} tempus_be_nox_analysis_tempus_driven_scalar_params.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "Transient;Analysis;Tempus;Scalar")
endif()

# RK 4 test
#set(testName ${testNameRoot}_Tempus_GERK)
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tempus_gerk.yaml
#               ${CMAKE_CURRENT_BINARY_DIR}/tempus_gerk.yaml COPYONLY)

#add_test(${testName} ${Albany.exe} tempus_gerk.yaml)
#set_tests_properties(${testName} PROPERTIES LABELS
#                                            "Basic;Tempus;Forward")
