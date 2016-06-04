macro(do_peridigm)

  message ("ctest state: BUILD_PERIDIGM")

  set_property (GLOBAL PROPERTY SubProject Peridigm)
  set_property (GLOBAL PROPERTY Label Peridigm)

  set (CONFIGURE_OPTIONS
    "-DCMAKE_BUILD_TYPE:STRING=Release"
    "-DENABLE_INSTALL:BOOL=ON"
    "-DCMAKE_INSTALL_PREFIX:PATH=${CTEST_BINARY_DIRECTORY}/PeridigmInstall"
    "-DTRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "-DCMAKE_C_COMPILER:STRING=mpicc"
    "-DCMAKE_CXX_COMPILER:STRING=mpicxx"
    "-DBOOST_ROOT=${BOOST_ROOT}"
    "-DUSE_DAKOTA:BOOL=OFF"
    "-DUSE_PV:BOOL=OFF"
    "-DUSE_PALS:BOOL=OFF"
    "-DCMAKE_CXX_FLAGS:STRING='-O2 -std=c++11 -Wall -pedantic -Wno-long-long -ftrapv -Wno-deprecated'"
    "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF")

  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/PeridigmBuild")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/PeridigmBuild)
  endif (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/PeridigmBuild")

  ctest_configure (
    BUILD "${CTEST_BINARY_DIRECTORY}/PeridigmBuild"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Peridigm"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    APPEND)

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure RETURN_VALUE S_HAD_ERROR)

    if (S_HAD_ERROR)
      message ("Cannot submit Peridigm configure results.")
    endif (S_HAD_ERROR)
  endif (CTEST_DO_SUBMIT)

  if (HAD_ERROR)
    message ("Cannot configure Peridigm build!")
    set (BUILD_PERIDIGM FALSE)
  endif (HAD_ERROR)

  if (BUILD_PERIDIGM)
    set (CTEST_BUILD_TARGET install)
    message ("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")
    ctest_build (
      BUILD "${CTEST_BINARY_DIRECTORY}/PeridigmBuild"
      RETURN_VALUE HAD_ERROR
      NUMBER_ERRORS BUILD_NUM_ERRORS
      APPEND)
    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Build RETURN_VALUE S_HAD_ERROR)
      if (S_HAD_ERROR)
        message ("Cannot submit Peridigm build results.")
      endif (S_HAD_ERROR)
    endif (CTEST_DO_SUBMIT)
    if (HAD_ERROR)
      message ("Cannot build Peridigm.")
      set (BUILD_PERIDIGM FALSE)
    endif (HAD_ERROR)
    if (BUILD_NUM_ERRORS GREATER 0)
      message ("Encountered build errors in Peridigm.")
      set (BUILD_PERIDIGM FALSE)
    endif (BUILD_NUM_ERRORS GREATER 0)
  endif (BUILD_PERIDIGM)

  message ("After configuring and building, BUILD_PERIDIGM = ${BUILD_PERIDIGM}")

end_macro(do_peridigm)
