macro(do_intel COMMON_CONFIGURE_OPTIONS BTYPE)

# Intel
  message ("ctest state: BUILD_${BTYPE}")

  set (LABLAS_LIBRARIES "-L$ENV{SIERRA_MKL_LIB_PATH} -Wl,--start-group $ENV{SIERRA_MKL_LIB_PATH}/libmkl_intel_lp64.a $ENV{SIERRA_MKL_LIB_PATH}/libmkl_core.a $ENV{SIERRA_MKL_LIB_PATH}/libmkl_sequential.a -Wl,--end-group")

  set (CONFIGURE_OPTIONS
    "${COMMON_CONFIGURE_OPTIONS}"
  "-DCMAKE_BUILD_TYPE:STRING=''"
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
#  "-DTPL_ENABLE_yaml-cpp:BOOL=ON"
#  "-Dyaml-cpp_INCLUDE_DIRS:PATH=${INTEL_PREFIX_DIR}/include"
#  "-Dyaml-cpp_LIBRARY_DIRS:PATH=${INTEL_PREFIX_DIR}/lib"
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
  "-DMPI_BIN_DIR:PATH=${MPI_BIN_DIR}"
  "-DMPI_EXEC:FILEPATH=${MPI_BIN_DIR}/mpiexec.hydra"
  "-DCMAKE_CXX_COMPILER:STRING=${MPI_BIN_DIR}/mpiicpc"
  "-DCMAKE_C_COMPILER:STRING=${MPI_BIN_DIR}/mpiicc"
  "-DCMAKE_Fortran_COMPILER:STRING=${MPI_BIN_DIR}/mpiifort"
  "-DTrilinos_EXTRA_LINK_FLAGS='-L${INTEL_PREFIX_DIR}/lib -lnetcdf -lpnetcdf -lhdf5_hl -lhdf5 -lifcore -lz -Wl,-rpath,${INTEL_PREFIX_DIR}/lib'"
  "-DCMAKE_AR:FILEPATH=xiar"
  "-DCMAKE_LINKER:FILEPATH=xild"
  "-DFC_FN_CASE=LOWER"
  "-DFC_FN_UNDERSCORE=UNDER"
  "-DCMAKE_INSTALL_PREFIX:PATH=${CTEST_INSTALL_DIRECTORY}/TrilinosIntelInstall"
  "-DTPL_BLAS_LIBRARIES:STRING=${LABLAS_LIBRARIES}"
  "-DTPL_LAPACK_LIBRARIES:STRING=${LABLAS_LIBRARIES}"
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

  if (CTEST_BUILD_CONFIGURATION MATCHES "Debug")
#   Set -g to enable retaining symbols
    set (CONFIGURE_OPTIONS
    "-DCMAKE_CXX_FLAGS:STRING='-g -O0 -mkl=sequential ${extra_cxx_flags}'"
    "-DCMAKE_C_FLAGS:STRING='-g -O0 -mkl=sequential'"
    "-DCMAKE_Fortran_FLAGS:STRING='-g -O0 -mkl=sequential'"
    "-DDART_TESTING_TIMEOUT:STRING=2400"
      "${CONFIGURE_OPTIONS}")
  else (CTEST_BUILD_CONFIGURATION MATCHES "Debug")

    set (CONFIGURE_OPTIONS
    "-DCMAKE_CXX_FLAGS:STRING='-xHost -O3 -fp-speculation=safe -DNDEBUG -mkl=sequential ${extra_cxx_flags}'"
    "-DCMAKE_C_FLAGS:STRING='-xHost -O3 -DNDEBUG -mkl=sequential'"
    "-DCMAKE_Fortran_FLAGS:STRING='-xHost -O3 -DNDEBUG -mkl=sequential'"
    "-DDART_TESTING_TIMEOUT:STRING=600"
      "${CONFIGURE_OPTIONS}")

#    "-DCMAKE_CXX_FLAGS:STRING='-axAVX -O3 -DNDEBUG -diag-disable=cpu-dispatch -mkl=sequential ${extra_cxx_flags}'"
#    "-DCMAKE_CXX_FLAGS:STRING='-O0 -g -diag-disable=cpu-dispatch -mkl=sequential ${extra_cxx_flags}'"
#    "-DCMAKE_C_FLAGS:STRING='-axAVX -O3 -diag-disable=cpu-dispatch -DNDEBUG -mkl=sequential'"
#    "-DCMAKE_C_FLAGS:STRING='-O0 -g -diag-disable=cpu-dispatch -mkl=sequential'"
#    "-DCMAKE_Fortran_FLAGS:STRING='-axAVX -O3 -DNDEBUG -diag-disable=cpu-dispatch -mkl=sequential'"
#    "-DCMAKE_Fortran_FLAGS:STRING='-O0 -g -diag-disable=cpu-dispatch -mkl=sequential'"

  endif (CTEST_BUILD_CONFIGURATION MATCHES "Debug")

# Clean up build area
  IF (CLEAN_BUILD)
    IF(EXISTS "${CTEST_BINARY_DIRECTORY}/${BTYPE}" )
      FILE(REMOVE_RECURSE "${CTEST_BINARY_DIRECTORY}/${BTYPE}")
    ENDIF()
  ENDIF()

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

# Clean up install area
    IF (CLEAN_BUILD)
      IF(EXISTS "${CTEST_INSTALL_DIRECTORY}/TrilinosIntelInstall" )
        FILE(REMOVE_RECURSE "${CTEST_INSTALL_DIRECTORY}/TrilinosIntelInstall")
      ENDIF()
    ENDIF()

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

# Run Trilinos tests 

    set (CTEST_TEST_TIMEOUT 600)
    CTEST_TEST(
      BUILD "${CTEST_BINARY_DIRECTORY}/${BTYPE}"
      #              PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
      #              INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
      #NUMBER_FAILED  TEST_NUM_FAILED
      RETURN_VALUE  HAD_ERROR
      )

    if (HAD_ERROR)
      message("Some Trilinos tests failed.")
    endif (HAD_ERROR)

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Test
      RETURN_VALUE  S_HAD_ERROR
      )

      if (S_HAD_ERROR)
        message ("Cannot submit Trilinos test results!")
      endif (S_HAD_ERROR)
    endif (CTEST_DO_SUBMIT)

  endif (BUILD_INTEL_TRILINOS)

if (BUILD_INTEL_ALBANY)
  message ("ctest state: BUILD_INTEL_ALBANY")

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${CTEST_INSTALL_DIRECTORY}/TrilinosIntelInstall"
    "-DENABLE_CONTACT:BOOL=OFF"
    "-DENABLE_LCM:BOOL=ON"
    "-DENABLE_LANDICE:BOOL=ON"
    "-DENABLE_BGL:BOOL=OFF"
    "-DENABLE_ATO:BOOL=OFF"
    "-DENABLE_ASCR:BOOL=OFF"
    "-DENABLE_AERAS:BOOL=ON"
    "-DENABLE_64BIT_INT:BOOL=ON"
    "-DENABLE_DEMO_PDES:BOOL=ON"
    "-DENABLE_STRONG_FPE_CHECK:BOOL=ON"
    "-DDISABLE_LCM_EXODIFF_SENSITIVE_TESTS:BOOL=ON"
   )

  if (BUILD_SCOREC)
    set (CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS}
      "-DENABLE_SCOREC:BOOL=ON")
  endif (BUILD_SCOREC)

# Clean up build area
   IF (CLEAN_BUILD)
     IF(EXISTS "${CTEST_BINARY_DIRECTORY}/AlbanyIntel" )
       FILE(REMOVE_RECURSE "${CTEST_BINARY_DIRECTORY}/AlbanyIntel")
     ENDIF()
   ENDIF()
  
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
