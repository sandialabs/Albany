if (ALBANY_SEACAS)
  # 1. Copy Input files from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_galerkin_exo.xml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_galerkin_exo.xml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_gaussnewton_exo.xml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_gaussnewton_exo.xml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_galerkin_trunc_exo.xml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_galerkin_trunc_exo.xml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_galerkin_trunc_colloc_exo.xml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_galerkin_trunc_colloc_exo.xml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_galerkin_trunc_colloc_sample_exo.xml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_galerkin_trunc_colloc_sample_exo.xml COPYONLY)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fullpodbasis.in.exo
                 ${CMAKE_CURRENT_BINARY_DIR}/fullpodbasis.in.exo COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fullpodbasis.in.exo.4.0
                 ${CMAKE_CURRENT_BINARY_DIR}/fullpodbasis.in.exo.4.0 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fullpodbasis.in.exo.4.1
                 ${CMAKE_CURRENT_BINARY_DIR}/fullpodbasis.in.exo.4.1 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fullpodbasis.in.exo.4.2
                 ${CMAKE_CURRENT_BINARY_DIR}/fullpodbasis.in.exo.4.2 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fullpodbasis.in.exo.4.3
                 ${CMAKE_CURRENT_BINARY_DIR}/fullpodbasis.in.exo.4.3 COPYONLY)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fullsampledbasis.in.exo
                 ${CMAKE_CURRENT_BINARY_DIR}/fullsampledbasis.in.exo COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fullsampledbasis.in.exo.4.0
                 ${CMAKE_CURRENT_BINARY_DIR}/fullsampledbasis.in.exo.4.0 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fullsampledbasis.in.exo.4.1
                 ${CMAKE_CURRENT_BINARY_DIR}/fullsampledbasis.in.exo.4.1 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fullsampledbasis.in.exo.4.2
                 ${CMAKE_CURRENT_BINARY_DIR}/fullsampledbasis.in.exo.4.2 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fullsampledbasis.in.exo.4.3
                 ${CMAKE_CURRENT_BINARY_DIR}/fullsampledbasis.in.exo.4.3 COPYONLY)

  # 2. Name the test with the directory name
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # 3. Create the test with this name and standard executable
  add_test(${testName}_galerkin_exo ${Albany.exe} input_galerkin_exo.xml)
  add_test(${testName}_gaussnewton_exo ${Albany.exe} input_gaussnewton_exo.xml)
  add_test(${testName}_galerkin_trunc_exo ${Albany.exe} input_galerkin_trunc_exo.xml)

# Currently failing in the Tpetra branch
  add_test(${testName}_galerkin_trunc_colloc_exo ${Albany.exe} input_galerkin_trunc_colloc_exo.xml)
  add_test(${testName}_galerkin_trunc_colloc_sample_exo ${Albany.exe} input_galerkin_trunc_colloc_sample_exo.xml)

endif (ALBANY_SEACAS)
