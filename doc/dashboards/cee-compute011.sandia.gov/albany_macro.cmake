macro(do_albany CONFIGURE_OPTIONS BTYPE)

  SET(CONFIG_SUCCESS FALSE)

  message ("ctest state: BUILD_${BTYPE}")

  set_property (GLOBAL PROPERTY SubProject ${BTYPE})
  set_property (GLOBAL PROPERTY Label ${BTYPE})

# Clean up build area
  IF (CLEAN_BUILD)
    IF(EXISTS "${CTEST_BINARY_DIRECTORY}/${BTYPE}" )
      FILE(REMOVE_RECURSE "${CTEST_BINARY_DIRECTORY}/${BTYPE}")
    ENDIF()
  ENDIF()

  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/${BTYPE}")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/${BTYPE})
  endif (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/${BTYPE}")

# We might eventually want to install albany on an nfs mounted filesystep
#   set (CONFIGURE_OPTIONS
#      "-DCMAKE_INSTALL_PREFIX:PATH=${CTEST_INSTALL_DIRECTORY}/${BTYPE}"
#      "${CONFIGURE_OPTIONS}")

  #
  # The build 
  #

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/${BTYPE}"
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
      message ("Cannot submit Albany configure results!")
    endif (S_HAD_ERROR)
  endif (CTEST_DO_SUBMIT)

  if (HAD_ERROR)
    message ("Cannot configure Albany build!")
  else(HAD_ERROR)
    SET(CONFIG_SUCCESS TRUE)
  endif (HAD_ERROR)

  #
  # Build Albany 
  #

  IF(CONFIG_SUCCESS)
  SET(BUILD_SUCCESS FALSE)

  set (CTEST_BUILD_TARGET all)

# We might eventually want to install albany on an nfs mounted filesystep
#  set (CTEST_BUILD_TARGET install)
# Clean up build area
#  IF (CLEAN_BUILD)
#    IF(EXISTS "${CTEST_INSTALL_DIRECTORY}/${BTYPE}" )
#      FILE(REMOVE_RECURSE "${CTEST_INSTALL_DIRECTORY}/${BTYPE}")
#    ENDIF()
#  ENDIF()

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
      message ("Cannot submit Albany build results!")
    endif (S_HAD_ERROR)
  endif (CTEST_DO_SUBMIT)

  if (HAD_ERROR)
    message ("Cannot build Albany!")
  else(HAD_ERROR)
    SET(BUILD_SUCCESS TRUE)
  endif (HAD_ERROR)

  if (BUILD_LIBS_NUM_ERRORS GREATER 0)
    message ("Encountered build errors in Albany 64 bit build. Exiting!")
    SET(BUILD_SUCCESS FALSE)
  endif (BUILD_LIBS_NUM_ERRORS GREATER 0)
  #
  # Run Albany tests
  #

  IF(BUILD_SUCCESS)

  CTEST_TEST(
    BUILD "${CTEST_BINARY_DIRECTORY}/${BTYPE}"
    #              PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
    #              INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
    #NUMBER_FAILED  TEST_NUM_FAILED
    RETURN_VALUE  HAD_ERROR
    )

  if (HAD_ERROR)
   message("Some Albany tests failed.")
  endif (HAD_ERROR)

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Test
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit Albany test results!")
    endif (S_HAD_ERROR)
  endif (CTEST_DO_SUBMIT)

  ENDIF(BUILD_SUCCESS)
  ENDIF(CONFIG_SUCCESS)

endmacro(do_albany CONFIGURE_OPTIONS BTYPE)
