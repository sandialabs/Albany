macro(do_trilinosclang11)

  message ("ctest state: BUILD_TRILINOSCLANG11")
  #
  # Configure the Trilinos/SCOREC Clang build
  #

  set_property (GLOBAL PROPERTY SubProject TrilinosClang++11)
  set_property (GLOBAL PROPERTY Label TrilinosClang++11)

  set (CONFIGURE_OPTIONS
    "${COMMON_CONFIGURE_OPTIONS}"
    "-DTPL_ENABLE_MPI:BOOL=ON"
    "-DMPI_BASE_DIR:PATH=${PREFIX_DIR}/clang"
    #
    "-DTrilinos_ENABLE_CXX11:BOOL=ON"
    "-DCMAKE_CXX_FLAGS:STRING='-Os -w -DNDEBUG ${extra_cxx_flags}'"
    "-DCMAKE_C_FLAGS:STRING='-Os -w -DNDEBUG'"
    "-DCMAKE_Fortran_FLAGS:STRING='-Os -w -DNDEBUG'"
    "-DTrilinos_ENABLE_SCOREC:BOOL=ON"
    "-DMDS_ID_TYPE:STRING='long long int'"
    "-DSCOREC_DISABLE_STRONG_WARNINGS:BOOL=ON"
    "-DTrilinos_EXTRA_LINK_FLAGS='-L${PREFIX_DIR}/lib -lhdf5_hl -lhdf5 -lz -lm'"
    "-DCMAKE_INSTALL_PREFIX:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstallC11"
    "-DBUILD_SHARED_LIBS:BOOL=OFF"
    "-DTPL_ENABLE_SuperLU:BOOL=OFF"
    "-DAmesos2_ENABLE_KLU2:BOOL=ON")

  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/TriBuildC11")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/TriBuildC11)
  endif (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/TriBuildC11")

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuildC11"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Trilinos"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure
      RETURN_VALUE  S_HAD_ERROR)

    if (S_HAD_ERROR)
      message ("Cannot submit TrilinosClang++11 configure results!")
    endif (S_HAD_ERROR)
  endif (CTEST_DO_SUBMIT)

  if (HAD_ERROR)
    message ("Cannot configure TrilinosClang++11 build!")
  endif (HAD_ERROR)

  #set (CTEST_BUILD_TARGET all)
  set (CTEST_BUILD_TARGET install)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuildC11"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Build
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit TrilinoClang++11 build results!")
    endif (S_HAD_ERROR)

  endif (CTEST_DO_SUBMIT)

  if (HAD_ERROR)
    message ("Cannot build Trilinos with Clang!")
  endif (HAD_ERROR)

  if (BUILD_LIBS_NUM_ERRORS GREATER 0)
    message ("Encountered build errors in Trilinos Clang build. Exiting!")
  endif (BUILD_LIBS_NUM_ERRORS GREATER 0)

endmacro(do_trilinosclang11)
