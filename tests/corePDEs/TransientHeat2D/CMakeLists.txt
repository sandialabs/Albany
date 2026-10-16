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
    # Transient optimization tests (AlbanyAnalysis, ROL), on the Heat 2D problem: 5x5 mesh, Backward Euler
    # with dt = 0.05 on [0,1], bound constraints, and an objective with optimum inside the bounds (except for
    # ActiveBound). The response is the misfit 100*||T - T*||^2 with an analytic target T* (Problem->Target
    # Temperature), at the final time (FinalTime), integrated in time (TimeIntegral), or summed over the times
    # 0.25, 0.5, 0.75 (SumOverTimes). The gradient is computed either by ROL with the dynamic constraints of the
    # time steps (ROL driver, discrete adjoint) or by Tempus with its adjoint sensitivities (Tempus driver,
    # continuous adjoint). Each test checks the gradient at the initial guess (output only), then optimizes.
    #  - FluxSource: flux controls on the left and right edges and a source; exact optimum (-1, 1, 0.5).
    #  - DirichletField: Dirichlet control on the bottom edge (distributed parameter dirichlet_field);
    #    exact optimum d = 1.
    #  - DirichletFieldReg: DirichletField with the regularization 0.5*|d|^2; the misfit is not zero at
    #    the optimum, which depends on the accuracy of the gradient. With the Tempus driver, the Dirichlet
    #    rows need the "Algebraic Equations Mass Diagonal" option of the Tempus adjoint (set by Albany).
    #  - ActiveBound: FluxSource with the source bounded by 0.3 < 0.5.
    # The expected optima come from an independent implementation of the discrete problems.
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/identity_6x6.mm
                   ${CMAKE_CURRENT_BINARY_DIR}/identity_6x6.mm COPYONLY)

    set(testName ${testNameRoot}_TransientOpt_FluxSource_ROL_FinalTime)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/transient_opt_flux_source_rol_final_time.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/transient_opt_flux_source_rol_final_time.yaml COPYONLY)
    add_test(${testName} ${AlbanyAnalysis.exe} transient_opt_flux_source_rol_final_time.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "Transient;Analysis;ROL;FinalTime;Scalar")

    set(testName ${testNameRoot}_TransientOpt_FluxSource_ROL_TimeIntegral)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/transient_opt_flux_source_rol_time_integral.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/transient_opt_flux_source_rol_time_integral.yaml COPYONLY)
    add_test(${testName} ${AlbanyAnalysis.exe} transient_opt_flux_source_rol_time_integral.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "Transient;Analysis;ROL;TimeIntegral;Scalar")

    set(testName ${testNameRoot}_TransientOpt_FluxSource_ROL_SumOverTimes)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/transient_opt_flux_source_rol_sum_over_times.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/transient_opt_flux_source_rol_sum_over_times.yaml COPYONLY)
    add_test(${testName} ${AlbanyAnalysis.exe} transient_opt_flux_source_rol_sum_over_times.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "Transient;Analysis;ROL;SumOverTimes;Scalar")

    set(testName ${testNameRoot}_TransientOpt_FluxSource_Tempus_FinalTime)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/transient_opt_flux_source_tempus_final_time.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/transient_opt_flux_source_tempus_final_time.yaml COPYONLY)
    add_test(${testName} ${AlbanyAnalysis.exe} transient_opt_flux_source_tempus_final_time.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "Transient;Analysis;Tempus;FinalTime;Scalar")

    set(testName ${testNameRoot}_TransientOpt_FluxSource_Tempus_TimeIntegral)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/transient_opt_flux_source_tempus_time_integral.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/transient_opt_flux_source_tempus_time_integral.yaml COPYONLY)
    add_test(${testName} ${AlbanyAnalysis.exe} transient_opt_flux_source_tempus_time_integral.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "Transient;Analysis;Tempus;TimeIntegral;Scalar")

    set(testName ${testNameRoot}_TransientOpt_FluxSource_Tempus_SumOverTimes)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/transient_opt_flux_source_tempus_sum_over_times.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/transient_opt_flux_source_tempus_sum_over_times.yaml COPYONLY)
    add_test(${testName} ${AlbanyAnalysis.exe} transient_opt_flux_source_tempus_sum_over_times.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "Transient;Analysis;Tempus;SumOverTimes;Scalar")

    set(testName ${testNameRoot}_TransientOpt_DirichletField_ROL_TimeIntegral)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/transient_opt_dirichlet_field_rol_time_integral.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/transient_opt_dirichlet_field_rol_time_integral.yaml COPYONLY)
    add_test(${testName} ${AlbanyAnalysis.exe} transient_opt_dirichlet_field_rol_time_integral.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "Transient;Analysis;ROL;TimeIntegral;Distributed")

    set(testName ${testNameRoot}_TransientOpt_DirichletField_ROL_SumOverTimes)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/transient_opt_dirichlet_field_rol_sum_over_times.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/transient_opt_dirichlet_field_rol_sum_over_times.yaml COPYONLY)
    add_test(${testName} ${AlbanyAnalysis.exe} transient_opt_dirichlet_field_rol_sum_over_times.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "Transient;Analysis;ROL;SumOverTimes;Distributed")

    set(testName ${testNameRoot}_TransientOpt_DirichletField_Tempus_FinalTime)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/transient_opt_dirichlet_field_tempus_final_time.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/transient_opt_dirichlet_field_tempus_final_time.yaml COPYONLY)
    add_test(${testName} ${AlbanyAnalysis.exe} transient_opt_dirichlet_field_tempus_final_time.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "Transient;Analysis;Tempus;FinalTime;Distributed")

    set(testName ${testNameRoot}_TransientOpt_DirichletField_Tempus_TimeIntegral)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/transient_opt_dirichlet_field_tempus_time_integral.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/transient_opt_dirichlet_field_tempus_time_integral.yaml COPYONLY)
    add_test(${testName} ${AlbanyAnalysis.exe} transient_opt_dirichlet_field_tempus_time_integral.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "Transient;Analysis;Tempus;TimeIntegral;Distributed")

    set(testName ${testNameRoot}_TransientOpt_DirichletField_Tempus_SumOverTimes)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/transient_opt_dirichlet_field_tempus_sum_over_times.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/transient_opt_dirichlet_field_tempus_sum_over_times.yaml COPYONLY)
    add_test(${testName} ${AlbanyAnalysis.exe} transient_opt_dirichlet_field_tempus_sum_over_times.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "Transient;Analysis;Tempus;SumOverTimes;Distributed")

    set(testName ${testNameRoot}_TransientOpt_DirichletFieldReg_ROL_FinalTime)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/transient_opt_dirichlet_field_reg_rol_final_time.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/transient_opt_dirichlet_field_reg_rol_final_time.yaml COPYONLY)
    add_test(${testName} ${AlbanyAnalysis.exe} transient_opt_dirichlet_field_reg_rol_final_time.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "Transient;Analysis;ROL;FinalTime;Distributed")

    set(testName ${testNameRoot}_TransientOpt_DirichletFieldReg_ROL_SumOverTimes)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/transient_opt_dirichlet_field_reg_rol_sum_over_times.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/transient_opt_dirichlet_field_reg_rol_sum_over_times.yaml COPYONLY)
    add_test(${testName} ${AlbanyAnalysis.exe} transient_opt_dirichlet_field_reg_rol_sum_over_times.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "Transient;Analysis;ROL;SumOverTimes;Distributed")

    set(testName ${testNameRoot}_TransientOpt_DirichletFieldReg_Tempus_FinalTime)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/transient_opt_dirichlet_field_reg_tempus_final_time.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/transient_opt_dirichlet_field_reg_tempus_final_time.yaml COPYONLY)
    add_test(${testName} ${AlbanyAnalysis.exe} transient_opt_dirichlet_field_reg_tempus_final_time.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "Transient;Analysis;Tempus;FinalTime;Distributed")

    set(testName ${testNameRoot}_TransientOpt_DirichletFieldReg_Tempus_SumOverTimes)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/transient_opt_dirichlet_field_reg_tempus_sum_over_times.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/transient_opt_dirichlet_field_reg_tempus_sum_over_times.yaml COPYONLY)
    add_test(${testName} ${AlbanyAnalysis.exe} transient_opt_dirichlet_field_reg_tempus_sum_over_times.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "Transient;Analysis;Tempus;SumOverTimes;Distributed")

    set(testName ${testNameRoot}_TransientOpt_ActiveBound_ROL_TimeIntegral)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/transient_opt_active_bound_rol_time_integral.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/transient_opt_active_bound_rol_time_integral.yaml COPYONLY)
    add_test(${testName} ${AlbanyAnalysis.exe} transient_opt_active_bound_rol_time_integral.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "Transient;Analysis;ROL;TimeIntegral;Scalar")
endif()

# RK 4 test
#set(testName ${testNameRoot}_Tempus_GERK)
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tempus_gerk.yaml
#               ${CMAKE_CURRENT_BINARY_DIR}/tempus_gerk.yaml COPYONLY)

#add_test(${testName} ${Albany.exe} tempus_gerk.yaml)
#set_tests_properties(${testName} PROPERTIES LABELS
#                                            "Basic;Tempus;Forward")
