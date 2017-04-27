cmake_minimum_required(VERSION 3.0.1)

set(LINE "------------------------------------------------------------")

function(do_config PKG_IN SOURCE_IN BUILD_IN OPTS_IN RETVAR)
  set(${RETVAR} FALSE PARENT_SCOPE)
  if (EXISTS "${BUILD_IN}/CMakeCache.txt")
    file(REMOVE "${BUILD_IN}/CMakeCache.txt")
  endif()
  if (EXISTS "${BUILD_IN}/CMakeFiles")
    file(REMOVE_RECURSE "${BUILD_IN}/CMakeFiles")
  endif()
  ctest_configure(
    BUILD "${BUILD_IN}"
    SOURCE "${SOURCE_IN}"
    OPTIONS "${OPTS_IN}"
    RETURN_VALUE CONFIG_ERR
    APPEND
  )
  if (CTEST_DO_SUBMIT)
    ctest_submit(PARTS Configure RETURN_VALUE SUBMIT_ERR)
    if(SUBMIT_ERR)
      message("Cannot submit ${PKG_IN} configure results!")
    endif()
  endif()
  if (CONFIG_ERR)
    message("Cannot configure ${PKG_IN}!")
    return()
  endif()
  set(${RETVAR} TRUE PARENT_SCOPE)
endfunction(do_config)

function(do_build PKG_IN BUILD_IN NPROCS_IN TARGET_IN RETVAR)
  set(${RETVAR} FALSE PARENT_SCOPE)
  message("BUILDING ${PKG_IN} ...")
  message("${LINE}")
  ctest_build(
    BUILD "${BUILD_IN}"
    APPEND
    FLAGS "-j ${NPROCS_IN}"
    TARGET "${TARGET_IN}"
    NUMBER_ERRORS NERRS
    NUMBER_WARNINGS NWARNS
    RETURN_VALUE STATUS
  )
  if (CTEST_DO_SUBMIT)
    ctest_submit(PARTS Build RETURN_VALUE SUBMIT_ERR)
    if(SUBMIT_ERR)
      message("Cannot submit ${PKG_IN} build results!")
    endif()
  endif()
  if (STATUS)
    string(TOUPPER "${TARGET_IN}" TARGET_ALLCAPS)
    message("*** MAKE ${TARGET_ALLCAPS} COMMAND FAILED ***")
    return()
  endif()
  set(${RETVAR} TRUE PARENT_SCOPE)
endfunction(do_build)

function(do_test BUILD_IN RETVAR)
  set(${RETVAR} FALSE PARENT_SCOPE)
  if (NOT EXISTS "${BUILD_IN}")
    message("Build directory does not exist. Run:")
    message("  [clean-]config-build.sh ${PKG_IN} ...")
    message("to create.")
    return()
  endif()
  message("TESTING ${PKG_IN} ...")
  message("${LINE}")
  ctest_test(
    BUILD "${BUILD_IN}"
    APPEND
    RETURN_VALUE ERR
  )
  if (CTEST_DO_SUBMIT)
    ctest_submit(PARTS Test RETURN_VALUE SUBMIT_ERR)
    if(SUBMIT_ERR)
      message("Cannot submit ${PKG_IN} test results!")
    endif()
  endif()
endfunction(do_test)

function(lcm_do_config PACKAGE PACKAGE_DIR INSTALL_DIR)
  if (${PACKAGE} STREQUAL "trilinos")
    set(OPTS
      "-DBUILD_SHARED_LIBS:BOOL=ON"
      "-DCMAKE_BUILD_TYPE:STRING=\"$ENV{BUILD_STRING}\""
      "-DCMAKE_CXX_COMPILER:FILEPATH=\"$ENV{MPI_BIN}/mpicxx\""
      "-DCMAKE_C_COMPILER:FILEPATH=\"$ENV{MPI_BIN}/mpicc\""
      "-DCMAKE_Fortran_COMPILER:FILEPATH=\"$ENV{MPI_BIN}/mpif90\""
      "-DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR}"
      "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF"
      "-DTPL_ENABLE_MPI:BOOL=ON"
      "-DTPL_ENABLE_BinUtils:BOOL=OFF"
      "-DTPL_MPI_INCLUDE_DIRS:STRING=\"$ENV{MPI_INC}\""
      "-DTPL_MPI_LIBRARY_DIRS:STRING=\"$ENV{MPI_LIB}\""
      "-DMPI_BIN_DIR:PATH=\"$ENV{MPI_BIN}\""
      "-DTPL_ENABLE_Boost:BOOL=ON"
      "-DTPL_ENABLE_BoostLib:BOOL=ON"
      "-DBoost_INCLUDE_DIRS:STRING=\"$ENV{BOOST_INC}\""
      "-DBoost_LIBRARY_DIRS:STRING=\"$ENV{BOOST_LIB}\""
      "-DBoostLib_INCLUDE_DIRS:STRING=\"$ENV{BOOSTLIB_INC}\""
      "-DBoostLib_LIBRARY_DIRS:STRING=\"$ENV{BOOSTLIB_LIB}\""
      "-DTPL_ENABLE_yaml-cpp:BOOL=ON"
      "-Dyaml-cpp_INCLUDE_DIRS:STRING=\"$ENV{YAML_CPP_INC}\""
      "-Dyaml-cpp_LIBRARY_DIRS:STRING=\"$ENV{YAML_CPP_LIB}\""
      "-DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF"
      "-DTrilinos_ENABLE_ALL_PACKAGES:BOOL=OFF"
      "-DTrilinos_ENABLE_CXX11:BOOL=ON"
      "-DTrilinos_ENABLE_EXAMPLES:BOOL=OFF"
      "-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON"
      "-DTrilinos_VERBOSE_CONFIGURE:BOOL=OFF"
      "-DTrilinos_WARNINGS_AS_ERRORS_FLAGS:STRING="""
      "-DTeuchos_ENABLE_STACKTRACE:BOOL=OFF"
      "-DTeuchos_ENABLE_DEFAULT_STACKTRACE:BOOL=OFF"
      "-DKokkos_ENABLE_CXX11:BOOL=ON"
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
      "-DIntrepid2_ENABLE_KokkosDynRankView:BOOL=ON"
      "-DMiniTensor_ENABLE_TESTS:BOOL=ON"
      "-DROL_ENABLE_TESTS:BOOL=OFF"
      "-DPhalanx_INDEX_SIZE_TYPE:STRING=\"$ENV{LCM_PHALANX_INDEX_TYPE}\""
      "-DPhalanx_KOKKOS_DEVICE_TYPE:STRING=\"$ENV{LCM_KOKKOS_DEVICE}\""
      "-DPhalanx_SHOW_DEPRECATED_WARNINGS:BOOL=OFF"
      "-DTpetra_ENABLE_Kokkos_Refactor:BOOL=ON"
      "-DTpetra_INST_PTHREAD:BOOL=$ENV{LCM_TPETRA_INST_PTHREAD}"
      "-DTPL_ENABLE_HDF5:BOOL=OFF"
      "-DTPL_ENABLE_HWLOC:STRING=$ENV{LCM_ENABLE_HWLOC}"
      "-DTPL_ENABLE_Matio:BOOL=OFF"
      "-DTPL_ENABLE_Netcdf:BOOL=ON"
      "-DTPL_ENABLE_X11:BOOL=OFF"
      "-DTPL_Netcdf_INCLUDE_DIRS:STRING=\"$ENV{LCM_NETCDF_INC}\""
      "-DTPL_Netcdf_LIBRARY_DIRS:STRING=\"$ENV{LCM_NETCDF_LIB}\""
      "-DTPL_Netcdf_LIBRARIES:STRING=\"$ENV{LCM_NETCDF_LIB}/libnetcdf.so\""
      "-DTPL_Netcdf_PARALLEL:BOOL=ON"
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
      "-DTrilinos_ENABLE_Phalanx:BOOL=ON"
      "-DTrilinos_ENABLE_Piro:BOOL=ON"
      "-DTrilinos_ENABLE_ROL:BOOL=ON"
      "-DTrilinos_ENABLE_Rythmos:BOOL=ON"
      "-DTrilinos_ENABLE_SEACAS:BOOL=ON"
      "-DTrilinos_ENABLE_STKClassic:BOOL=OFF"
      "-DTrilinos_ENABLE_STKIO:BOOL=ON"
      "-DTrilinos_ENABLE_STKMesh:BOOL=ON"
      "-DTrilinos_ENABLE_Sacado:BOOL=ON"
      "-DTrilinos_ENABLE_Shards:BOOL=ON"
      "-DTrilinos_ENABLE_Stokhos:BOOL=ON"
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
    )
    if (ENV{LCM_SLFAD_SIZE})
      set(OPTS ${OPTS} $ENV{LCM_SLFAD_SIZE})
    endif()
    set(EXTRA_REPOS)
    if (EXISTS "${PACKAGE_DIR}/DataTransferKit")
      set(EXTRA_REPOS ${EXTRA_REPOS} DataTransferKit)
      set(OPTS ${OPTS}
        "-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=OFF"
        "-DTrilinos_ENABLE_DataTransferKit:BOOL=ON"
        "-DDataTransferKit_ENABLE_DBC:BOOL=ON"
        "-DDataTransferKit_ENABLE_TESTS:BOOL=OFF"
        "-DDataTransferKit_ENABLE_EXAMPLES:BOOL=OFF"
        "-DTPL_ENABLE_MOAB:BOOL=OFF"
        "-DTPL_ENABLE_Libmesh:BOOL=OFF"
        )
    else()
      set(OPTS ${OPTS} "-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON"
    endif()
    if (EXTRA_REPOS)
      string(REPLACE ";" "," EXTRA_REPOS "${EXTRA_REPOS}")
      set(OPTS ${OPTS} "-DTrilinos_EXTRA_REPOSITORIES:STRING=\"${EXTRA_REPOS}\"")
    endif()
  elseif (${LCM_PACKAGE} STREQUAL "albany")
  endif()
endfunction(lcm_do_config)

function(lcm_env_single SCRIPT_NAME PACKAGE NUM_PROCS)
  if (NOT ("${PACKAGE}" MATCHES "trilinos|albany"))
    message(FATAL_ERROR "Specify package [trilinos|albany]")
  endif()
  if (NOT ("$ENV{ARCH}" MATCHES "serial|openmp|pthreads|cuda"))
    message(FATAL_ERROR "Specify architecture [serial|openmp|pthreads|cuda]")
  endif()
  if (NOT ("$ENV{TOOL_CHAIN}" MATCHES "gcc|clang|intel|pgi"))
    message(FATAL_ERROR "Specify tool chain [gcc|clang|intel|pgi]")
  endif()
  if (NOT ("$ENV{BUILD_TYPE}" MATCHES "debug|release|profile|small|mixed"))
    message(FATAL_ERROR "Specify build type [debug|release|profile|small|mixed]")
  endif()
  if (NOT ENV{NUM_PROCS})
    set(ENV{NUM_PROCS} 1)
  endif()

  if (PACKAGE STREQUAL "trilinos")
    set(PACKAGE_STRING "TRILINOS")
    set(PACKAGE_NAME "Trilinos")
  elseif (PACKAGE STREQUAL "albany")
    set(PACKAGE_STRING "ALBANY")
    set(PACKAGE_NAME "Albany")
  endif()

  set(PACKAGE_DIR "$ENV{LCM_DIR}/${PACKAGE_NAME}")
  set(INSTALL_DIR "$ENV{LCM_DIR}/trilinos-install-$ENV{BUILD}")
  set(BUILD_DIR "$ENV{LCM_DIR}/${PACKAGE}-build-$ENV{BUILD}")
  set(PREFIX "${PACKAGE}-$ENV{BUILD}")
  execute_process(COMMAND hostname OUTPUT_VARIABLE HOST)
  
  set(PACKAGE_STRING ${PACKAGE_STRING} PARENT_SCOPE)
  set(PACKAGE_NAME ${PACKAGE_NAME} PARENT_SCOPE)
  set(PACKAGE_DIR ${PACKAGE_DIR} PARENT_SCOPE)
  set(INSTALL_DIR ${INSTALL_DIR} PARENT_SCOPE)
  set(BUILD_DIR ${BUILD_DIR} PARENT_SCOPE)
  set(PREFIX ${PREFIX} PARENT_SCOPE)
  set(HOST ${HOST} PARENT_SCOPE)
endfunction(lcm_env_single)

function(lcm_check_script_name SCRIPT_NAME)
  set(expected_script_name "")
  set(steps clean config build test)
  foreach(step IN LISTS steps)
    if (SCRIPT_NAME MATCHES "${step}")
      if (expected_script_name STREQUAL "")
        set(expected_script_name "${step}")
      else()
        set(expected_script_name "${expected_script_name}-${step}")
      endif()
    endif()
  endforeach()
  set(expected_script_name "${expected_script_name}.sh")
  if (NOT (SCRIPT_NAME STREQUAL expected_script_name))
    message(FATAL_ERROR "Unrecognized script name: ${SCRIPT_NAME}. Expected ${expected_script_name}")
  endif()
endfunction()

function(lcm_main SCRIPT_NAME PACKAGE NUM_PROCS)
  lcm_check_script_name("${SCRIPT_NAME}")
  lcm_env_single("${SCRIPT_NAME}" "${PACKAGE}" "${NUM_PROCS}")
  set(CTEST_TEST_TYPE Nightly)
  message("${LINE}")
  message("${PACKAGE_NAME} directory\t: ${PACKAGE_DIR}")
  message("Install directory \t: ${INSTALL_DIR}")
  message("Build directory\t\t: ${BUILD_DIR}")
  message("${LINE}"
endfunction()

lcm_main("${SCRIPT_NAME}" "${PACKAGE}" "${NUM_PROCS}")
