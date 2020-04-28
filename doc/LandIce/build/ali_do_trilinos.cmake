if(LCM_DO_TRILINOS_CMAKE)
  return()
endif()
set(LCM_DO_TRILINOS_CMAKE true)

include(${CMAKE_CURRENT_LIST_DIR}/snl_helpers.cmake)

function(ali_do_trilinos)
  set(BOOL_OPTS
      "CLEAN_BUILD"
      "CLEAN_INSTALL"
      "DO_UPDATE"
      "DO_CONFIG"
      "DO_BUILD"
      "DO_TEST"
     )
  set(UNARY_OPTS
      "BUILD_THREADS"
      "RESULT_VARIABLE"
      "BUILD_ID_STRING"
    )
  message("ali_do_trilinos(${ARGN})")
  cmake_parse_arguments(ARG "${BOOL_OPTS}" "${UNARY_OPTS}" "" ${ARGN}) 
  if (ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
        "ali_do_trilinos called with unrecognized arguments ${ARG_UNPARSED_ARGUMENTS}")
  endif()
  set(CONFIG_OPTS
      "-DBUILD_SHARED_LIBS:BOOL=ON"
      "-DCMAKE_BUILD_TYPE:STRING=$ENV{BUILD_STRING}"
      "-DCMAKE_CXX_COMPILER:FILEPATH=$ENV{MPI_BIN}/mpicxx"
      "-DCMAKE_C_COMPILER:FILEPATH=$ENV{MPI_BIN}/mpicc"
      "-DCMAKE_Fortran_COMPILER:FILEPATH=$ENV{MPI_BIN}/mpif90"
      "-DCMAKE_INSTALL_PREFIX:PATH=$ENV{TEST_DIR}/trilinos-install-${ARG_BUILD_ID_STRING}"
      "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF"
      "-DTPL_ENABLE_MPI:BOOL=ON"
      "-DTPL_ENABLE_BinUtils:BOOL=OFF"
      "-DTPL_MPI_INCLUDE_DIRS:STRING=$ENV{MPI_INC}"
      "-DTPL_MPI_LIBRARY_DIRS:STRING=$ENV{MPI_LIB}"
      "-DMPI_BIN_DIR:PATH=$ENV{MPI_BIN}"
      "-DTPL_ENABLE_Boost:BOOL=ON"
      "-DTPL_ENABLE_BoostLib:BOOL=ON"
      "-DBoost_INCLUDE_DIRS:STRING=$ENV{BOOST_INC}"
      "-DBoost_LIBRARY_DIRS:STRING=$ENV{BOOST_LIB}"
      "-DBoostLib_INCLUDE_DIRS:STRING=$ENV{BOOSTLIB_INC}"
      "-DBoostLib_LIBRARY_DIRS:STRING=$ENV{BOOSTLIB_LIB}"
      "-DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF"
      "-DTrilinos_ENABLE_ALL_PACKAGES:BOOL=OFF"
      "-DTrilinos_ENABLE_CXX11:BOOL=ON"
      "-DTrilinos_ENABLE_EXAMPLES:BOOL=OFF"
      "-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON"
      "-DTrilinos_VERBOSE_CONFIGURE:BOOL=OFF"
      "-DTrilinos_WARNINGS_AS_ERRORS_FLAGS:STRING="
      "-DTeuchos_ENABLE_STACKTRACE:BOOL=OFF"
      "-DTeuchos_ENABLE_DEFAULT_STACKTRACE:BOOL=OFF"
      "-DKokkos_ENABLE_Cuda_UVM:BOOL=$ENV{LCM_ENABLE_UVM}"
      "-DKokkos_ENABLE_EXAMPLES:BOOL=$ENV{LCM_ENABLE_KOKKOS_EXAMPLES}"
      "-DKokkos_ENABLE_OpenMP:BOOL=$ENV{LCM_ENABLE_OPENMP}"
      "-DKokkos_ENABLE_Pthread:BOOL=$ENV{LCM_ENABLE_PTHREADS}"
      "-DKokkos_ENABLE_Serial:BOOL=ON"
      "-DKokkos_ENABLE_TESTS:BOOL=OFF"
      "-DTPL_ENABLE_CUDA:STRING=$ENV{LCM_ENABLE_CUDA}"
      "-DTPL_ENABLE_CUSPARSE:BOOL=$ENV{LCM_ENABLE_CUSPARSE}"
      "-DAmesos2_ENABLE_KLU2:BOOL=ON"
      "-DEpetraExt_USING_HDF5:BOOL=OFF"
      "-DROL_ENABLE_TESTS:BOOL=OFF"
      "-DPhalanx_INDEX_SIZE_TYPE:STRING=$ENV{LCM_PHALANX_INDEX_TYPE}"
      "-DPhalanx_KOKKOS_DEVICE_TYPE:STRING=$ENV{LCM_KOKKOS_DEVICE}"
      "-DPhalanx_SHOW_DEPRECATED_WARNINGS:BOOL=OFF"
      "-DTpetra_ENABLE_Kokkos_Refactor:BOOL=ON"
      "-DTpetra_INST_PTHREAD:BOOL=$ENV{LCM_TPETRA_INST_PTHREAD}"
      "-DTPL_ENABLE_HDF5:BOOL=OFF"
      "-DTPL_ENABLE_HWLOC:STRING=$ENV{LCM_ENABLE_HWLOC}"
      "-DTPL_ENABLE_Matio:BOOL=OFF"
      "-DTPL_ENABLE_Netcdf:BOOL=ON"
      "-DTPL_ENABLE_X11:BOOL=OFF"
      "-DTPL_Netcdf_INCLUDE_DIRS:STRING=$ENV{NETCDF_INC}"
      "-DTPL_Netcdf_LIBRARY_DIRS:STRING=$ENV{NETCDF_LIB}"
      "-DTPL_Netcdf_LIBRARIES:STRING=$ENV{NETCDF_LIB}/libnetcdf.so"
      "-DTPL_Netcdf_PARALLEL:BOOL=$ENV{LCM_NETCDF_PARALLEL}"
      "-DTrilinos_ENABLE_Amesos2:BOOL=ON"
      "-DTrilinos_ENABLE_Amesos:BOOL=ON"
      "-DTrilinos_ENABLE_Anasazi:BOOL=ON"
      "-DTrilinos_ENABLE_AztecOO:BOOL=ON"
      "-DTrilinos_ENABLE_Belos:BOOL=ON"
      "-DTrilinos_ENABLE_EXAMPLES:BOOL=OFF"
      "-DTrilinos_ENABLE_Epetra:BOOL=ON"
      "-DTrilinos_ENABLE_EpetraExt:BOOL=ON"
      "-DTrilinos_ENABLE_Ifpack2:BOOL=ON"
      "-DTrilinos_ENABLE_Ifpack:BOOL=ON"
      "-DTrilinos_ENABLE_Intrepid2:BOOL=ON"
      "-DTrilinos_ENABLE_Kokkos:BOOL=ON"
      "-DTrilinos_ENABLE_KokkosAlgorithms:BOOL=ON"
      "-DTrilinos_ENABLE_KokkosContainers:BOOL=ON"
      "-DTrilinos_ENABLE_KokkosCore:BOOL=ON"
      "-DTrilinos_ENABLE_KokkosExample:BOOL=OFF"
      "-DTrilinos_ENABLE_MiniTensor:BOOL=ON"
      "-DTrilinos_ENABLE_ML:BOOL=ON"
      "-DTrilinos_ENABLE_MueLu:BOOL=ON"
      "-DTrilinos_ENABLE_NOX:BOOL=ON"
      "-DTrilinos_ENABLE_OpenMP:BOOL=$ENV{LCM_ENABLE_OPENMP}"
      "-DTrilinos_ENABLE_Pamgen:BOOL=ON"
      "-DTrilinos_ENABLE_PanzerExprEval:BOOL=ON"
      "-DTrilinos_ENABLE_Phalanx:BOOL=ON"
      "-DTrilinos_ENABLE_Piro:BOOL=ON"
      "-DTrilinos_ENABLE_ROL:BOOL=ON"
      "-DTrilinos_ENABLE_Rythmos:BOOL=ON"
      "-DTrilinos_ENABLE_SEACAS:BOOL=ON"
      "-DTrilinos_ENABLE_SEACASAprepro_lib:BOOL=ON"
      "-DTrilinos_ENABLE_STKClassic:BOOL=OFF"
      "-DTrilinos_ENABLE_STKIO:BOOL=ON"
      "-DTrilinos_ENABLE_STKMesh:BOOL=ON"
      "-DTrilinos_ENABLE_STKExprEval:BOOL=ON"
      "-DTrilinos_ENABLE_Sacado:BOOL=ON"
      "-DTrilinos_ENABLE_Shards:BOOL=ON"
      "-DTrilinos_ENABLE_Stokhos:BOOL=OFF"
      "-DTrilinos_ENABLE_Stratimikos:BOOL=ON"
      "-DTrilinos_ENABLE_TESTS:BOOL=OFF"
      "-DTrilinos_ENABLE_Teko:BOOL=ON"
      "-DTrilinos_ENABLE_Tempus:BOOL=ON"
      "-DTrilinos_ENABLE_Teuchos:BOOL=ON"
      "-DTrilinos_ENABLE_ThreadPool:BOOL=ON"
      "-DTrilinos_ENABLE_Thyra:BOOL=ON"
      "-DTrilinos_ENABLE_Tpetra:BOOL=ON"
      "-DTrilinos_ENABLE_Zoltan2:BOOL=ON"
      "-DTrilinos_ENABLE_Zoltan:BOOL=ON"
      "-DTempus_ENABLE_TESTS:BOOL=OFF"
      "-DTempus_ENABLE_EXAMPLES:BOOL=OFF"
      "-DTempus_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON"
      "-DPhalanx_ALLOW_MULTIPLE_EVALUATORS_FOR_SAME_FIELD:BOOL=ON"
      "-DTpetra_ENABLE_DEPRECATED_CODE:BOOL=OFF"
      "-DXpetra_ENABLE_DEPRECATED_CODE:BOOL=OFF"
      )
  if (DEFINED ENV{LCM_SLFAD_SIZE})
    set(CONFIG_OPTS ${CONFIG_OPTS} $ENV{LCM_SLFAD_SIZE})
  endif()
  if (DEFINED ENV{LCM_LINK_FLAGS})
    set(CONFIG_OPTS ${CONFIG_OPTS}
        "-DCMAKE_EXE_LINKER_FLAGS:STRING=$ENV{LCM_LINK_FLAGS}"
        "-DCMAKE_SHARED_LINKER_FLAGS:STRING=$ENV{LCM_LINK_FLAGS}"
       )
  endif()
  set(EXTRA_REPOS)
  set(SOURCE_DIR "$ENV{TEST_DIR}/Trilinos")
  if (EXISTS "${SOURCE_DIR}/DataTransferKit")
    set(EXTRA_REPOS ${EXTRA_REPOS} DataTransferKit)
    set(CONFIG_OPTS ${CONFIG_OPTS}
      "-DTpetra_INST_INT_UNSIGNED_LONG:BOOL=ON"
      "-DTrilinos_ENABLE_DataTransferKit:BOOL=ON"
      "-DDataTransferKit_ENABLE_DBC:BOOL=ON"
      "-DDataTransferKit_ENABLE_TESTS:BOOL=OFF"
      "-DDataTransferKit_ENABLE_EXAMPLES:BOOL=OFF"
      "-DTPL_ENABLE_MOAB:BOOL=OFF"
      "-DTPL_ENABLE_Libmesh:BOOL=OFF"
      )
  else()
    set(CONFIG_OPTS ${CONFIG_OPTS}
      "-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON")
  endif()
  if (EXTRA_REPOS)
    string(REPLACE ";" "," EXTRA_REPOS "${EXTRA_REPOS}")
    set(CONFIG_OPTS ${CONFIG_OPTS}
        "-DTrilinos_EXTRA_REPOSITORIES:STRING=${EXTRA_REPOS}")
  endif()
  set(ARG_BOOL_OPTS)
  foreach (BOOL_OPT IN LISTS BOOL_OPTS)
    if (ARG_${BOOL_OPT})
      if ("${BOOL_OPT}" STREQUAL "DO_BUILD")
        set(ARG_BOOL_OPTS ${ARG_BOOL_OPTS} "DO_INSTALL")
      else()
        set(ARG_BOOL_OPTS ${ARG_BOOL_OPTS} ${BOOL_OPT})
      endif()
    endif()
  endforeach()
  snl_do_subproject(${ARG_BOOL_OPTS}
      DO_PROJECT
      "PROJECT" "Albany"
      SOURCE_DIR "$ENV{TEST_DIR}/Trilinos"
      BUILD_DIR "$ENV{TEST_DIR}/trilinos-build-${ARG_BUILD_ID_STRING}"
      INSTALL_DIR "$ENV{TEST_DIR}/trilinos-install-${ARG_BUILD_ID_STRING}"
      CONFIG_OPTS "${CONFIG_OPTS}"
      BUILD_THREADS "${ARG_BUILD_THREADS}"
      RESULT_VARIABLE ERR
      )
  if (ARG_RESULT_VARIABLE)
    set(${ARG_RESULT_VARIABLE} ${ERR} PARENT_SCOPE)
  endif()
endfunction(ali_do_trilinos)
