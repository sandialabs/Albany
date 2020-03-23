macro(do_peridigm)

  message ("ctest state: BUILD_PERIDIGM")


  set (CONFIGURE_OPTIONS
    "-DTRILINOS_DIR:PATH=${CTEST_INSTALL_DIRECTORY}/TrilinosInstall/lib/cmake/Trilinos"
    "-DCMAKE_INSTALL_PREFIX:PATH=${CTEST_INSTALL_DIRECTORY}/PeridigmInstall"
    "-DCMAKE_BUILD_TYPE:STRING=Release"
    "-DENABLE_INSTALL:BOOL=ON"
    "-DUSE_DAKOTA:BOOL=OFF"
    "-DUSE_PV:BOOL=OFF"
    "-DBOOST_ROOT=${BOOST_ROOT}"
    "-DCMAKE_C_COMPILER:STRING=mpicc"
    "-DCMAKE_CXX_COMPILER:STRING=mpicxx"
    "-DCMAKE_CXX_FLAGS:STRING='-O3 -std=c++11 -march=native'"
    "-DCMAKE_CXX_LINK_FLAGS:STRING='-L${PREFIX_DIR}/lib -lhdf5_hl -lnetcdf -lboost_timer -lboost_chrono -Wl,-rpath,${PREFIX_DIR}/lib:${MKLHOME}:${INTEL_DIR}/lib/intel64'"
    )

# Clean up build area
  IF (CLEAN_BUILD)
    IF(EXISTS "${CTEST_BINARY_DIRECTORY}/PeridigmBuild" )
      FILE(REMOVE_RECURSE "${CTEST_BINARY_DIRECTORY}/PeridigmBuild")
    ENDIF()
  ENDIF()

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

# Clean up install area
    IF (CLEAN_BUILD)
        IF(EXISTS "${CTEST_INSTALL_DIRECTORY}/PeridigmInstall" )
          FILE(REMOVE_RECURSE "${CTEST_INSTALL_DIRECTORY}/PeridigmInstall")
        ENDIF()
    ENDIF()

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

# Copy the targets file where it should go
  configure_file(${CTEST_BINARY_DIRECTORY}/PeridigmBuild/peridigm-targets.cmake
                 ${CTEST_INSTALL_DIRECTORY}/PeridigmInstall/lib/Peridigm/cmake/peridigm-targets.cmake COPYONLY)

endmacro(do_peridigm)
