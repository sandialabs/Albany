macro(do_intel COMMON_CONFIGURE_OPTIONS BTYPE)

# Intel
  message ("ctest state: BUILD_${BTYPE}")
  set_property (GLOBAL PROPERTY SubProject ${BTYPE})
  set_property (GLOBAL PROPERTY Label ${BTYPE})

  set (LABLAS_LIBRARIES "-L${MKL_PATH}/mkl/lib/intel64 -Wl,--start-group ${MKL_PATH}/mkl/lib/intel64/libmkl_intel_lp64.a ${MKL_PATH}/mkl/lib/intel64/libmkl_core.a ${MKL_PATH}/mkl/lib/intel64/libmkl_sequential.a -Wl,--end-group")

  set (CONFIGURE_OPTIONS
    "${COMMON_CONFIGURE_OPTIONS}"
  "-DTPL_ENABLE_Pthread:BOOL=OFF"
  "-DBoost_INCLUDE_DIRS:PATH=${INTEL_BOOST_ROOT}/include"
  "-DBoost_LIBRARY_DIRS:PATH=${INTEL_BOOST_ROOT}/lib"
  "-DBoostLib_INCLUDE_DIRS:PATH=${INTEL_BOOST_ROOT}/include"
  "-DBoostLib_LIBRARY_DIRS:PATH=${INTEL_BOOST_ROOT}/lib"
  "-DBoostAlbLib_INCLUDE_DIRS:PATH=${INTEL_BOOST_ROOT}/include"
  "-DBoostAlbLib_LIBRARY_DIRS:PATH=${INTEL_BOOST_ROOT}/lib"
#
  "-DTPL_ENABLE_Netcdf:BOOL=ON"
  "-DNetcdf_INCLUDE_DIRS:PATH=${INTEL_PREFIX_DIR}/include"
  "-DNetcdf_LIBRARY_DIRS:PATH=${INTEL_PREFIX_DIR}/lib"
  "-DTPL_Netcdf_PARALLEL:BOOL=ON"
  "-DTPL_ENABLE_Pnetcdf:BOOL=ON"
  "-DPnetcdf_INCLUDE_DIRS:PATH=${INTEL_PREFIX_DIR}/include"
  "-DPnetcdf_LIBRARY_DIRS=${INTEL_PREFIX_DIR}/lib"
  #
  "-DTPL_ENABLE_HDF5:BOOL=ON"
  "-DHDF5_INCLUDE_DIRS:PATH=${INTEL_PREFIX_DIR}/include"
  "-DHDF5_LIBRARY_DIRS:PATH=${INTEL_PREFIX_DIR}/lib"
  "-DHDF5_LIBRARY_NAMES:STRING='hdf5_hl\\;hdf5\\;z'"
  #
  "-DTPL_ENABLE_Zlib:BOOL=ON"
  "-DZlib_INCLUDE_DIRS:PATH=${INTEL_PREFIX_DIR}/include"
  "-DZlib_LIBRARY_DIRS:PATH=${INTEL_PREFIX_DIR}/lib"
  #
  "-DTPL_ENABLE_yaml-cpp:BOOL=ON"
  "-Dyaml-cpp_INCLUDE_DIRS:PATH=${INTEL_PREFIX_DIR}/include"
  "-Dyaml-cpp_LIBRARY_DIRS:PATH=${INTEL_PREFIX_DIR}/lib"
  #
  "-DTPL_ENABLE_ParMETIS:BOOL=ON"
  "-DParMETIS_INCLUDE_DIRS:PATH=${INTEL_PREFIX_DIR}/include"
  "-DParMETIS_LIBRARY_DIRS:PATH=${INTEL_PREFIX_DIR}/lib"
  #
  "-DTPL_ENABLE_SuperLU:BOOL=ON"
  "-DSuperLU_INCLUDE_DIRS:PATH=${INTEL_PREFIX_DIR}/SuperLU_4.3/include"
  "-DSuperLU_LIBRARY_DIRS:PATH=${INTEL_PREFIX_DIR}/SuperLU_4.3/lib"
#
    "-DTPL_ENABLE_MPI:BOOL=ON"
    "-DMPI_BASE_DIR:PATH=${INTEL_MPI_DIR}"
    "-DMPI_BIN_DIR:PATH=${INTEL_MPI_DIR}/bin64"
    "-DMPI_EXEC:FILEPATH=${INTEL_MPI_DIR}/bin64/mpiexec.hydra"
    "-DCMAKE_CXX_COMPILER:STRING=${INTEL_MPI_DIR}/bin64/mpiicpc"
#    "-DCMAKE_CXX_FLAGS:STRING='-axAVX -O3 -DNDEBUG -diag-disable=cpu-dispatch -mkl=sequential ${extra_cxx_flags}'"
    "-DCMAKE_CXX_FLAGS:STRING='-O1 -DNDEBUG -diag-disable=cpu-dispatch -mkl=sequential ${extra_cxx_flags}'"
    "-DCMAKE_C_COMPILER:STRING=${INTEL_MPI_DIR}/bin64/mpiicc"
#    "-DCMAKE_C_FLAGS:STRING='-axAVX -O3 -diag-disable=cpu-dispatch -DNDEBUG -mkl=sequential'"
    "-DCMAKE_C_FLAGS:STRING='-O1 -diag-disable=cpu-dispatch -DNDEBUG -mkl=sequential'"
    "-DCMAKE_Fortran_COMPILER:STRING=${INTEL_MPI_DIR}/bin64/mpiifort"
#    "-DCMAKE_Fortran_FLAGS:STRING='-axAVX -O3 -DNDEBUG -diag-disable=cpu-dispatch -mkl=sequential'"
    "-DCMAKE_Fortran_FLAGS:STRING='-O1 -DNDEBUG -diag-disable=cpu-dispatch -mkl=sequential'"
    "-DTrilinos_EXTRA_LINK_FLAGS='-L${INTEL_PREFIX_DIR}/lib -lnetcdf -lpnetcdf -lhdf5_hl -lhdf5 -lifcore -lz -Wl,-rpath,${INTEL_PREFIX_DIR}/lib:${INTEL_DIR}/lib/intel64'"
    "-DCMAKE_AR:FILEPATH=${INTEL_DIR}/bin/intel64/xiar"
    "-DCMAKE_LINKER:FILEPATH=${INTEL_DIR}/linux/bin/intel64/xild"
    "-DCMAKE_INSTALL_PREFIX:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosIntelInstall"
    "-DTPL_BLAS_LIBRARIES:STRING=${LABLAS_LIBRARIES}"
    "-DTPL_LAPACK_LIBRARIES:STRING=${LABLAS_LIBRARIES}"
#    "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON"
#    "-DCMAKE_INSTALL_RPATH:STRING=${INTEL_MPI_DIR}/lib64;${INTEL_PREFIX_DIR}/lib;${INTEL_DIR}/lib/intel64"
    )

  if (BUILD_SCOREC)
    set (CONFIGURE_OPTIONS
      "-DTrilinos_ENABLE_SCOREC:BOOL=ON"
      "-DSCOREC_DISABLE_STRONG_WARNINGS:BOOL=ON"
      "-DTrilinos_ENABLE_EXPORT_MAKEFILES:BOOL=OFF"
      "-DTrilinos_ASSERT_MISSING_PACKAGES:BOOL=OFF"
      "-DZoltan_ENABLE_ULONG_IDS:Bool=ON"
      "${CONFIGURE_OPTIONS}")
  endif (BUILD_SCOREC)

  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/${BTYPE}")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/${BTYPE})
  endif (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/${BTYPE}")

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/${BTYPE}"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Trilinos"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR)

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure RETURN_VALUE S_HAD_ERROR)

    if (S_HAD_ERROR)
      message ("Cannot submit Trilinos/SCOREC configure results.")
    endif (S_HAD_ERROR)
  endif (CTEST_DO_SUBMIT)

  if (HAD_ERROR)
    message ("Cannot configure TrilinosIntel build.")
    set (BUILD_INTEL_ALBANY FALSE) # no need to go further, Albany needs Trilinos
    set (BUILD_INTEL_TRILINOS FALSE)
  endif (HAD_ERROR)

  if (BUILD_INTEL_TRILINOS)
    set (CTEST_BUILD_TARGET install)

    message ("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

    ctest_build (
      BUILD "${CTEST_BINARY_DIRECTORY}/${BTYPE}"
      RETURN_VALUE HAD_ERROR
      NUMBER_ERRORS BUILD_LIBS_NUM_ERRORS
      APPEND)

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Build RETURN_VALUE S_HAD_ERROR)

      if (S_HAD_ERROR)
        message ("Cannot submit TrilinosIntel build results.")
      endif (S_HAD_ERROR)

    endif (CTEST_DO_SUBMIT)

    if (HAD_ERROR)
      message ("Cannot build Trilinos.")
      set (BUILD_INTEL_ALBANY FALSE)
    endif (HAD_ERROR)

    if (BUILD_LIBS_NUM_ERRORS GREATER 0)
      message ("Encountered build errors in Trilinos build. Exiting.")
      set (BUILD_INTEL_ALBANY FALSE)
    endif (BUILD_LIBS_NUM_ERRORS GREATER 0)
  endif (BUILD_INTEL_TRILINOS)

if (BUILD_INTEL_ALBANY)
  message ("ctest state: BUILD_INTEL_ALBANY")
  set_property (GLOBAL PROPERTY SubProject AlbanyIntel)
  set_property (GLOBAL PROPERTY Label AlbanyIntel)

  # Copy from the Intel Trilinos block. Not actually needed here in practice,
  # but if I do debugging on this script, it's nice to be able to run just this
  # block without the Trilinos one.
#  set (ENV{LM_LICENSE_FILE} 7500@sitelicense.sandia.gov)
#  set (ENV{PATH}
#    ${INTEL_DIR}/compilers_and_libraries/linux/bin/intel64:${PATH})
#  set (ENV{LD_LIBRARY_PATH}
#    ${INTEL_DIR}/compilers_and_libraries/linux/lib/intel64:${INTEL_MPI_DIR}/lib:${INITIAL_LD_LIBRARY_PATH})

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosIntelInstall"
    "-DENABLE_LCM:BOOL=ON"
    "-DENABLE_MOR:BOOL=ON"
    "-DENABLE_FELIX:BOOL=ON"
    "-DENABLE_HYDRIDE:BOOL=ON"
    "-DENABLE_BGL:BOOL=OFF"
    "-DENABLE_AMP:BOOL=OFF"
    "-DENABLE_ATO:BOOL=ON"
    "-DENABLE_QCAD:BOOL=ON"
    "-DENABLE_ENSEMBLE:BOOL=OFF"
    "-DENABLE_SG:BOOL=OFF"
    "-DENABLE_ASCR:BOOL=OFF"
    "-DENABLE_AERAS:BOOL=ON"
    "-DENABLE_64BIT_INT:BOOL=OFF"
    "-DENABLE_DEMO_PDES:BOOL=ON"
    "-DENABLE_CHECK_FPE:BOOL=ON"
    "-DENABLE_LAME:BOOL=${USE_LAME}"
    "-DLAME_INCLUDE_DIR:PATH=${LAME_INC_DIR}"
    "-DLAME_LIBRARY_DIR:PATH=${LAME_LIB_DIR}"
    "-DLAME_LIBRARIES:PATH=${LAME_LIBRARIES}"
    "-DMATH_TOOLKIT_INCLUDE_DIR:PATH=${MATH_TOOLKIT_INC_DIR}"
    "-DMATH_TOOLKIT_LIBRARY_DIR:PATH=${MATH_TOOLKIT_LIB_DIR}")
  if (BUILD_SCOREC)
    set (CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS}
      "-DENABLE_SCOREC:BOOL=ON"
      "-DENABLE_GOAL:BOOL=ON")
  endif (BUILD_SCOREC)
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/AlbanyIntel")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/AlbanyIntel)
  endif (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/AlbanyIntel")

  CTEST_CONFIGURE (
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbanyIntel"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    APPEND)

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure RETURN_VALUE S_HAD_ERROR)
    
    if (S_HAD_ERROR)
      message ("Cannot submit Albany configure results.")
    endif (S_HAD_ERROR)
  endif (CTEST_DO_SUBMIT)

  if (HAD_ERROR)
    message ("Cannot configure Albany build.")
    set (BUILD_INTEL_ALBANY FALSE)
  endif (HAD_ERROR)
endif (BUILD_INTEL_ALBANY)

  if (BUILD_INTEL_ALBANY)
    set (CTEST_BUILD_TARGET all)

    message ("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

    CTEST_BUILD (
      BUILD "${CTEST_BINARY_DIRECTORY}/AlbanyIntel"
      RETURN_VALUE HAD_ERROR
      NUMBER_ERRORS BUILD_LIBS_NUM_ERRORS
      APPEND)

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Build
        RETURN_VALUE S_HAD_ERROR)

      if (S_HAD_ERROR)
        message ("Cannot submit Albany build results.")
      endif (S_HAD_ERROR)
    endif (CTEST_DO_SUBMIT)

    if (HAD_ERROR)
      message ("Cannot build Albany.")
      set (BUILD_INTEL_ALBANY FALSE)
    endif (HAD_ERROR)

    if (BUILD_LIBS_NUM_ERRORS GREATER 0)
      message ("Encountered build errors in Albany build.")
      set (BUILD_INTEL_ALBANY FALSE)
    endif (BUILD_LIBS_NUM_ERRORS GREATER 0)
  endif (BUILD_INTEL_ALBANY)

  if (BUILD_INTEL_ALBANY)
    #set (CTEST_TEST_TIMEOUT 120)
    CTEST_TEST (
      BUILD "${CTEST_BINARY_DIRECTORY}/AlbanyIntel"
      RETURN_VALUE HAD_ERROR)

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Test RETURN_VALUE S_HAD_ERROR)

      if (S_HAD_ERROR)
        message ("Cannot submit Albany test results.")
      endif (S_HAD_ERROR)
    endif (CTEST_DO_SUBMIT)
  endif (BUILD_INTEL_ALBANY)

endmacro(do_intel COMMON_CONFIGURE_OPTIONS BTYPE)
