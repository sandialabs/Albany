macro(do_albfunctor)

  message ("ctest state: BUILD_ALBFUNCTOR")
  # ALBANY_KOKKOS_UNDER_DEVELOPMENT build

  set_property (GLOBAL PROPERTY SubProject AlbanyFunctorDev)
  set_property (GLOBAL PROPERTY Label AlbanyFunctorDev)

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "-DENABLE_LCM:BOOL=ON"
    "-DENABLE_MOR:BOOL=ON"
    "-DENABLE_FELIX:BOOL=ON"
    "-DENABLE_HYDRIDE:BOOL=ON"
    "-DENABLE_AMP:BOOL=OFF"
    "-DENABLE_ATO:BOOL=ON"
    "-DENABLE_QCAD:BOOL=ON"
    "-DENABLE_ENSEMBLE:BOOL=OFF"
    "-DENABLE_SG:BOOL=OFF"
    "-DENABLE_ASCR:BOOL=OFF"
    "-DENABLE_AERAS:BOOL=ON"
    "-DENABLE_64BIT_INT:BOOL=OFF"
    "-DENABLE_DEMO_PDES:BOOL=ON"
    "-DENABLE_KOKKOS_UNDER_DEVELOPMENT:BOOL=ON"
    "-DENABLE_CHECK_FPE:BOOL=ON")
  if (BUILD_SCOREC)
    set (CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS}
      "-DENABLE_SCOREC:BOOL=ON"
      "-DENABLE_GOAL:BOOL=ON")
  endif (BUILD_SCOREC)
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/AlbanyFunctorDev")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/AlbanyFunctorDev)
  endif (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/AlbanyFunctorDev")

  CTEST_CONFIGURE (
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbanyFunctorDev"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    APPEND)

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure RETURN_VALUE S_HAD_ERROR)
    
    if (S_HAD_ERROR)
      message ("Cannot submit Albany configure results!")
      set (BUILD_ALBFUNCTOR FALSE)
    endif (S_HAD_ERROR)
  endif (CTEST_DO_SUBMIT)

  if (HAD_ERROR)
    message ("Cannot configure Albany build!")
    set (BUILD_ALBFUNCTOR FALSE)
  endif (HAD_ERROR)

  if (BUILD_ALBFUNCTOR)
    set (CTEST_BUILD_TARGET all)

    message ("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

    CTEST_BUILD (
      BUILD "${CTEST_BINARY_DIRECTORY}/AlbanyFunctorDev"
      RETURN_VALUE HAD_ERROR
      NUMBER_ERRORS BUILD_LIBS_NUM_ERRORS
      APPEND)

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Build
        RETURN_VALUE S_HAD_ERROR)

      if (S_HAD_ERROR)
        message ("Cannot submit Albany build results!")
        set (BUILD_ALBFUNCTOR FALSE)
      endif (S_HAD_ERROR)
    endif (CTEST_DO_SUBMIT)

    if (HAD_ERROR)
      message ("Cannot build Albany!")
      set (BUILD_ALBFUNCTOR FALSE)
    endif (HAD_ERROR)

    if (BUILD_LIBS_NUM_ERRORS GREATER 0)
      message ("Encountered build errors in Albany build.")
      set (BUILD_ALBFUNCTOR FALSE)
    endif (BUILD_LIBS_NUM_ERRORS GREATER 0)
  endif (BUILD_ALBFUNCTOR)

  if (BUILD_ALBFUNCTOR)
    set (CTEST_TEST_TIMEOUT 180)
    CTEST_TEST (
      BUILD "${CTEST_BINARY_DIRECTORY}/AlbanyFunctorDev"
      RETURN_VALUE HAD_ERROR)

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Test RETURN_VALUE S_HAD_ERROR)

      if (S_HAD_ERROR)
        message ("Cannot submit Albany test results!")
      endif (S_HAD_ERROR)
    endif (CTEST_DO_SUBMIT)
  endif (BUILD_ALBFUNCTOR)

endmacro(do_albfunctor)
