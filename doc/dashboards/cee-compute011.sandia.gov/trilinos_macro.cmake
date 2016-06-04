macro(do_trilinos COM_CONF)

  message ("ctest state: BUILD_TRILINOS")

  #
  # Configure the Trilinos/SCOREC build
  #

  set_property (GLOBAL PROPERTY SubProject Trilinos)
  set_property (GLOBAL PROPERTY Label Trilinos)

  set (CONFIGURE_OPTIONS
    "-DTPL_ENABLE_MPI:BOOL=ON"
    "-DMPI_BASE_DIR:PATH=${GCC_MPI_DIR}"
    "-DCMAKE_CXX_FLAGS:STRING='-O3 -march=native -w -DNDEBUG ${extra_cxx_flags}'"
    "-DCMAKE_C_FLAGS:STRING='-O3 -march=native -w -DNDEBUG'"
    "-DCMAKE_Fortran_FLAGS:STRING='-O3 -march=native -w -DNDEBUG'"
    "-DTrilinos_EXTRA_LINK_FLAGS='-L${PREFIX_DIR}/lib -lhdf5_hl -lhdf5 -lz -lm'"
    "-DCMAKE_INSTALL_PREFIX:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "${COM_CONF}"
    )

  if (BUILD_SCOREC)
    set (CONFIGURE_OPTIONS
      "-DTrilinos_ENABLE_SCOREC:BOOL=ON"
      "-DSCOREC_DISABLE_STRONG_WARNINGS:BOOL=ON"      
      "${CONFIGURE_OPTIONS}")
  endif (BUILD_SCOREC)

  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/TriBuild")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/TriBuild)
  endif (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/TriBuild")

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuild"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Trilinos"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit Trilinos/SCOREC configure results!")
    endif (S_HAD_ERROR)
  endif (CTEST_DO_SUBMIT)

  if (HAD_ERROR)
    message ("Cannot configure Trilinos/SCOREC build!")
  endif (HAD_ERROR)

  if (BUILD_SCOREC)
    #
    # SCOREC tools build inside Trilinos
    #
    # Note that we do a trick here, and just build the SCOREC_libs target, as we
    # build SCOREC as a Trilinos packages and its not possible to do that
    # independent of Trilinos. So, while this builds most of SCOREC, other
    # Trilinos capabilities are also built here.
    #

    set_property (GLOBAL PROPERTY SubProject SCOREC)
    set_property (GLOBAL PROPERTY Label SCOREC)
    set (CTEST_BUILD_TARGET "SCOREC_libs")

    MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

    CTEST_BUILD(
      BUILD "${CTEST_BINARY_DIRECTORY}/TriBuild"
      RETURN_VALUE  HAD_ERROR
      NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
      )

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Build
        RETURN_VALUE  S_HAD_ERROR
        )

      if (S_HAD_ERROR)
        message ("Cannot submit SCOREC build results!")
        set (BUILD_SCOREC FALSE)
      endif (S_HAD_ERROR)
    endif (CTEST_DO_SUBMIT)

    if (HAD_ERROR)
      message ("Cannot build SCOREC!")
      set (BUILD_SCOREC FALSE)
    endif (HAD_ERROR)

    if (BUILD_LIBS_NUM_ERRORS GREATER 0)
      message ("Encountered build errors in SCOREC build. Exiting!")
      set (BUILD_SCOREC FALSE)
    endif (BUILD_LIBS_NUM_ERRORS GREATER 0)
  endif (BUILD_SCOREC)

  #
  # Trilinos
  #
  # Build the rest of Trilinos and install everything
  #

  set_property (GLOBAL PROPERTY SubProject Trilinos)
  set_property (GLOBAL PROPERTY Label Trilinos)
  #set (CTEST_BUILD_TARGET all)
  set (CTEST_BUILD_TARGET install)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuild"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Build
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit Trilinos/SCOREC build results!")
    endif (S_HAD_ERROR)

  endif (CTEST_DO_SUBMIT)

  if (HAD_ERROR)
    message ("Cannot build Trilinos!")
  endif (HAD_ERROR)

  if (BUILD_LIBS_NUM_ERRORS GREATER 0)
    message ("Encountered build errors in Trilinos build. Exiting!")
  endif (BUILD_LIBS_NUM_ERRORS GREATER 0)

endmacro(do_trilinos COM_CONF)
