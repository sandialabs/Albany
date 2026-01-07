IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_IFPACK2)
  # Name the test with the directory name
  get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
  get_filename_component(parentDirName ${parentPath} NAME)
  get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  set (testName ${parentDirName}_${dirName})

  # 1d test
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_1d_stk.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_1d_stk.yaml COPYONLY)
  add_test(${testName}_1d_stk ${SerialAlbany.exe} input_1d_stk.yaml)
  set_tests_properties(${testName}_1d_stk PROPERTIES LABELS "Demo;Forward;Serial")

  # 1d test with adaptivity
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_1d_adapt_stk.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_1d_adapt_stk.yaml COPYONLY)
  add_test(${testName}_1d_adapt_stk ${SerialAlbany.exe} input_1d_adapt_stk.yaml)
  set_tests_properties(${testName}_1d_adapt_stk PROPERTIES LABELS "Demo;Forward;Serial;Adapt")

  # 2d test
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_2d_stk.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_2d_stk.yaml COPYONLY)
  add_test(${testName}_2d_stk ${SerialAlbany.exe} input_2d_stk.yaml)
  set_tests_properties(${testName}_2d_stk PROPERTIES LABELS "Demo;Forward;Serial")

  # If omegah is available, repeat all tests but with Omega_h as discretization lib
  if (ENABLE_OMEGAH)
    # 1d test
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_1d_omegah.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_1d_omegah.yaml COPYONLY)
    add_test(${testName}_1d_omegah ${SerialAlbany.exe} input_1d_omegah.yaml)
    set_tests_properties(${testName}_1d_omegah PROPERTIES LABELS "Demo;Forward;Serial")

    # 1d test with adaptivity
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_1d_adapt_omegah.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_1d_adapt_omegah.yaml COPYONLY)
    add_test(${testName}_1d_adapt_omega_h ${SerialAlbany.exe} input_1d_adapt_omegah.yaml)
    set_tests_properties(${testName}_1d_adapt_omega_h PROPERTIES LABELS "Demo;Forward;Serial;Adapt")

    set(Albany2.exe ${MPIEX} ${MPIPRE} ${MPINPF} 2 ${MPIPOST} ${AlbanyPath})
    add_test(${testName}_1d_adapt_omega_h_2proc ${Albany2.exe} input_1d_adapt_omegah.yaml)
    set_tests_properties(${testName}_1d_adapt_omega_h_2proc PROPERTIES LABELS "Demo;Forward;Adapt")

    # 2d test
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_2d_omegah.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_2d_omegah.yaml COPYONLY)
    add_test(${testName}_2d_omegah ${SerialAlbany.exe} input_2d_omegah.yaml)
    set_tests_properties(${testName}_2d_omegah PROPERTIES LABELS "Demo;Forward;Serial")

    if (ENABLE_MESHFIELDS)
      # 2d test with adaptivity
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_2d_adapt_omegah.yaml
                     ${CMAKE_CURRENT_BINARY_DIR}/input_2d_adapt_omegah.yaml COPYONLY)
      set(testBaseName ${testName}_2d_adapt_omegah)
      add_test(${testBaseName}_Serial ${SerialAlbany.exe} input_2d_adapt_omegah.yaml)
      set_tests_properties(${testBaseName}_Serial PROPERTIES LABELS "Demo;Forward;Serial;Adapt")

      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_2d_adapt_omegah_2proc.yaml
                     ${CMAKE_CURRENT_BINARY_DIR}/input_2d_adapt_omegah_2proc.yaml COPYONLY)
      add_test(${testBaseName}_2proc ${Albany2.exe} input_2d_adapt_omegah_2proc.yaml)
      set_tests_properties(${testBaseName}_2proc PROPERTIES LABELS "Demo;Forward;Adapt")

      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_2d_adapt_omegah_2proc_small.yaml
                     ${CMAKE_CURRENT_BINARY_DIR}/input_2d_adapt_omegah_2proc_small.yaml COPYONLY)
      set(Albany2.exe ${MPIEX} ${MPIPRE} ${MPINPF} 2 ${MPIPOST} ${AlbanyPath})
      add_test(${testBaseName}_2proc_small ${Albany2.exe} input_2d_adapt_omegah_2proc_small.yaml)
      set_tests_properties(${testBaseName}_2proc_small PROPERTIES LABELS "Demo;Forward;Adapt")
    ENDIF()

  ENDIF()

ENDIF ()
