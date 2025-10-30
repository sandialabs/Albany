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
    add_test(${testName}_1d_adapt_Omega_h ${SerialAlbany.exe} input_1d_adapt_omegah.yaml)
    set_tests_properties(${testName}_1d_adapt_omega_h PROPERTIES LABELS "Demo;Forward;Serial;Adapt")

    # 2d test
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_2d_omegah.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_2d_omegah.yaml COPYONLY)
    add_test(${testName}_2d_omegah ${SerialAlbany.exe} input_2d_omegah.yaml)
    set_tests_properties(${testName}_2d_omegah PROPERTIES LABELS "Demo;Forward;Serial")

    # 2d test with adaptivity
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_2d_adapt_omegah.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_2d_adapt_omegah.yaml COPYONLY)
    add_test(${testName}_2d_adapt_omegah ${SerialAlbany.exe} input_2d_adapt_omegah.yaml)
    set_tests_properties(${testName}_2d_adapt_omegah PROPERTIES LABELS "Demo;Forward;Serial;Adapt")

  ENDIF()

ENDIF ()
