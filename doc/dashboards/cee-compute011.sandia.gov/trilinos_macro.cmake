macro(do_trilinos CONFIGURE_OPTIONS BTYPE)

  message ("ctest state: BUILD_${BTYPE}")

  #
  # Configure the Trilinos/SCOREC build
  #

  set_property (GLOBAL PROPERTY SubProject ${BTYPE})
  set_property (GLOBAL PROPERTY Label ${BTYPE})

  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/${BTYPE}")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/${BTYPE})
  endif (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/${BTYPE}")

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/${BTYPE}"
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

  SET(SEPARATE_BUILD_SCOREC FALSE)

  if (SEPARATE_BUILD_SCOREC)
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
      BUILD "${CTEST_BINARY_DIRECTORY}/${BTYPE}"
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

  endif (SEPARATE_BUILD_SCOREC)

  #
  # Trilinos
  #
  # Build the rest of Trilinos and install everything
  #

  set_property (GLOBAL PROPERTY SubProject ${BTYPE})
  set_property (GLOBAL PROPERTY Label ${BTYPE})
  #set (CTEST_BUILD_TARGET all)
  set (CTEST_BUILD_TARGET install)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/${BTYPE}"
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

endmacro(do_trilinos CONFIGURE_OPTIONS BTYPE)
