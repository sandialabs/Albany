include(AlbanyCreateTest)

IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_IFPACK2)
  # Name the test with the directory name
  get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
  get_filename_component(parentDirName ${parentPath} NAME)
  get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  set (testName ${parentDirName}_${dirName})

  # 1d test
  AlbanyCreateTest(${testName}_1d_stk input_1d_stk.yaml
    SERIAL LABELS Demo Forward Serial)

  # 1d test with adaptivity
  AlbanyCreateTest(${testName}_1d_adapt_stk input_1d_adapt_stk.yaml
    SERIAL LABELS Demo Forward Serial Adapt)

  # 2d test
  AlbanyCreateTest(${testName}_2d_stk input_2d_stk.yaml
    SERIAL LABELS Demo Forward Serial)

  # If omegah is available, repeat all tests but with Omega_h as discretization lib
  if (ENABLE_OMEGAH)
    AlbanyCreateTest(${testName}_1d_omegah input_1d_omegah.yaml
      SERIAL LABELS Demo Forward Serial)

    # 1d test with adaptivity
    AlbanyCreateTest(${testName}_1d_adapt_omegah input_1d_adapt_omegah.yaml
      SERIAL LABELS Demo Forward Serial Adapt)

    # 2d test
    AlbanyCreateTest(${testName}_2d_omegah input_2d_omegah.yaml
      SERIAL LABELS Demo Forward Serial)

    if (ENABLE_MESHFIELDS)
      # 2d test with adaptivity
      AlbanyCreateTest(${testName}_2d_adapt_omegah input_2d_adapt_omegah.yaml
        SERIAL LABELS Demo Forward Serial Adapt)

      AlbanyCreateTest(${testName}_2d_adapt_omegah_2proc input_2d_adapt_omegah_2proc.yaml
        LABELS Demo Forward Serial Adapt)
      set(Albany2.exe ${MPIEX} ${MPIPRE} ${MPINPF} 2 ${MPIPOST} ${AlbanyPath})
      add_test(${testBaseName}_2proc ${Albany2.exe} input_2d_adapt_omegah_2proc.yaml)
      set_tests_properties(${testBaseName}_2proc PROPERTIES LABELS "Demo;Forward;Adapt")
    ENDIF()

  ENDIF()

ENDIF ()
