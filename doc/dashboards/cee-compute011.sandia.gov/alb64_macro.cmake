macro(do_alb64)

  message ("ctest state: BUILD_ALB64")

  set_property (GLOBAL PROPERTY SubProject Albany64Bit)
  set_property (GLOBAL PROPERTY Label Albany64Bit)

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "-DENABLE_64BIT_INT:BOOL=ON"
    "-DENABLE_ALBANY_EPETRA_EXE:BOOL=OFF"
    "-DENABLE_LCM:BOOL=ON"
    "-DENABLE_LCM_SPECULATIVE:BOOL=OFF"
    "-DENABLE_HYDRIDE:BOOL=ON"
    "-DENABLE_ENSEMBLE:BOOL=OFF"
    "-DENABLE_SG:BOOL=OFF"
    "-DENABLE_QCAD:BOOL=OFF"
    "-DENABLE_MOR:BOOL=OFF"
    "-DENABLE_CHECK_FPE:BOOL=ON")
  if (BUILD_SCOREC)
    set (CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS}
      "-DENABLE_SCOREC:BOOL=ON"
      "-DENABLE_GOAL:BOOL=ON")
  endif (BUILD_SCOREC)

  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/Albany64Bit")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/Albany64Bit)
  endif (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/Albany64Bit")

  #
  # The 64 bit build 
  #

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/Albany64Bit"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    APPEND
    )

  # Read the CTestCustom.cmake file to turn off ignored tests

  #CTEST_READ_CUSTOM_FILES("${CTEST_BINARY_DIRECTORY}/AlbanyT64")

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit Albany 64 bit configure results!")
    endif (S_HAD_ERROR)
  endif (CTEST_DO_SUBMIT)

  if (HAD_ERROR)
    message ("Cannot configure Albany 64 bit build!")
  endif (HAD_ERROR)

  #
  # Build Albany 64 bit
  #

  set (CTEST_BUILD_TARGET all)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/Albany64Bit"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Build
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit Albany 64 bit build results!")
    endif (S_HAD_ERROR)
  endif (CTEST_DO_SUBMIT)

  if (HAD_ERROR)
    message ("Cannot build Albany 64 bit!")
  endif (HAD_ERROR)

  if (BUILD_LIBS_NUM_ERRORS GREATER 0)
    message ("Encountered build errors in Albany 64 bit build. Exiting!")
  endif (BUILD_LIBS_NUM_ERRORS GREATER 0)
  #
  # Run Albany 64 bit tests
  #

  CTEST_TEST(
    BUILD "${CTEST_BINARY_DIRECTORY}/Albany64Bit"
    #              PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
    #              INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
    #NUMBER_FAILED  TEST_NUM_FAILED
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Test
      RETURN_VALUE  HAD_ERROR
      )

    if (HAD_ERROR)
      message ("Cannot submit Albany 64 bit test results!")
    endif (HAD_ERROR)
  endif (CTEST_DO_SUBMIT)

endmacro(do_alb64)
