macro(do_alb32)

  message ("ctest state: BUILD_ALB32")

  # Configure the Albany 32 Bit build 
  # Builds everything!

  set_property (GLOBAL PROPERTY SubProject Albany32Bit)
  set_property (GLOBAL PROPERTY Label Albany32Bit)

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "-DENABLE_LCM:BOOL=ON"
    "-DENABLE_CONTACT:BOOL=ON"
    "-DENABLE_LCM_SPECULATIVE:BOOL=OFF"
    "-DENABLE_HYDRIDE:BOOL=ON"
    "-DENABLE_ENSEMBLE:BOOL=OFF"
    "-DENABLE_SG:BOOL=OFF"
    "-DENABLE_FELIX:BOOL=ON"
    "-DENABLE_AERAS:BOOL=ON"
    "-DENABLE_QCAD:BOOL=ON"
    "-DENABLE_MOR:BOOL=ON"
    "-DENABLE_ATO:BOOL=ON"
    "-DENABLE_AMP:BOOL=OFF"
    "-DENABLE_ASCR:BOOL=OFF"
    "-DENABLE_CHECK_FPE:BOOL=ON")
  if (BUILD_SCOREC)
    set (CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS}
      "-DENABLE_SCOREC:BOOL=ON"
      "-DENABLE_GOAL:BOOL=ON")
  endif (BUILD_SCOREC)
  if (BUILD_PERIDIGM)
    set (CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS}
      "-DENABLE_PERIDIGM:BOOL=ON"
      "-DPeridigm_DIR:PATH=${CTEST_BINARY_DIRECTORY}/PeridigmInstall/lib/Peridigm/cmake")
  endif (BUILD_PERIDIGM)
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/Albany32Bit")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/Albany32Bit)
  endif (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/Albany32Bit")

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/Albany32Bit"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit Albany configure results!")
    endif (S_HAD_ERROR)
  endif (CTEST_DO_SUBMIT)

  if (HAD_ERROR)
    message ("Cannot configure Albany build!")
  endif (HAD_ERROR)

  #
  # Build Albany
  #

  set (CTEST_BUILD_TARGET all)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/Albany32Bit"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Build
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit Albany build results!")
    endif (S_HAD_ERROR)
  endif (CTEST_DO_SUBMIT)

  if (HAD_ERROR)
    message ("Cannot build Albany!")
  endif (HAD_ERROR)

  if (BUILD_LIBS_NUM_ERRORS GREATER 0)
    message ("Encountered build errors in Albany build. Exiting!")
  endif (BUILD_LIBS_NUM_ERRORS GREATER 0)

  #
  # Run Albany tests
  #

  CTEST_TEST(
    BUILD "${CTEST_BINARY_DIRECTORY}/Albany32Bit"
    #              PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
    #              INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
    #NUMBER_FAILED  TEST_NUM_FAILED
    RETURN_VALUE  HAD_ERROR
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Test
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit Albany test results!")
    endif (S_HAD_ERROR)
  endif (CTEST_DO_SUBMIT)

  #if (HAD_ERROR)
  #	message(FATAL_ERROR "Some Albany tests failed.")
  #endif (HAD_ERROR)

endmacro(do_alb32)
