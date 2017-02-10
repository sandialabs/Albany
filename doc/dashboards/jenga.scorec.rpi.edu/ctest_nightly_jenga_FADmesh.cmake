cmake_minimum_required(VERSION 2.8)

SET(CTEST_DO_SUBMIT ON)
SET(CTEST_TEST_TYPE Nightly)

#SET(CTEST_DO_SUBMIT OFF)
#SET(CTEST_TEST_TYPE Experimental)

# What to build and test
SET(BUILD_TRILINOS TRUE)
SET(BUILD_ALB32 TRUE)
SET(BUILD_ALB64 FALSE)

SET(BUILD_TRILINOSCLANG FALSE)
SET(BUILD_ALB64CLANG FALSE)
SET(BUILD_ALBFUNCTOR FALSE)

SET(DOWNLOAD TRUE)
SET(CLEAN_BUILD TRUE)

# Begin User inputs:
set( CTEST_SITE             "jenga.scorec.rpi.edu" ) # generally the output of hostname
set( CTEST_DASHBOARD_ROOT   "$ENV{TEST_DIRECTORY}" ) # writable path
set( CTEST_SCRIPT_DIRECTORY   "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set( CTEST_CMAKE_GENERATOR  "Unix Makefiles" ) # What is your compilation apps ?
set( CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?

set(INITIAL_LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})

set( CTEST_PROJECT_NAME         "Albany" )
set( CTEST_SOURCE_NAME          repos)
set( CTEST_BUILD_NAME           "linux-gcc-${CTEST_BUILD_CONFIGURATION}")
set( CTEST_BINARY_NAME          build)

SET(PREFIX_DIR /users/ghansen)

SET (CTEST_SOURCE_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_SOURCE_NAME}")
SET (CTEST_BINARY_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_BINARY_NAME}")

IF(NOT EXISTS "${CTEST_SOURCE_DIRECTORY}")
  FILE(MAKE_DIRECTORY "${CTEST_SOURCE_DIRECTORY}")
ENDIF()
IF(NOT EXISTS "${CTEST_BINARY_DIRECTORY}")
  FILE(MAKE_DIRECTORY "${CTEST_BINARY_DIRECTORY}")
ENDIF()

configure_file(${CTEST_SCRIPT_DIRECTORY}/CTestConfig.cmake
  ${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake COPYONLY)

SET(CTEST_NIGHTLY_START_TIME "00:00:00 UTC")
SET (CTEST_CMAKE_COMMAND "${PREFIX_DIR}/bin/cmake")
SET (CTEST_COMMAND "${PREFIX_DIR}/bin/ctest -D ${CTEST_TEST_TYPE}")
SET (CTEST_BUILD_FLAGS "-j8")

SET(CTEST_DROP_METHOD "http")

IF (CTEST_DROP_METHOD STREQUAL "http")
  SET(CTEST_DROP_SITE "my.cdash.com")
  SET(CTEST_PROJECT_NAME "Albany")
  SET(CTEST_DROP_LOCATION "/submit.php?project=Albany")
  SET(CTEST_TRIGGER_SITE "")
  SET(CTEST_DROP_SITE_CDASH TRUE)
ENDIF()

find_program(CTEST_GIT_COMMAND NAMES git)
find_program(CTEST_SVN_COMMAND NAMES svn)

SET(Trilinos_REPOSITORY_LOCATION https://github.com/trilinos/trilinos.git)

SET(Albany_REPOSITORY_LOCATION https://github.com/gahansen/Albany.git)

IF (CLEAN_BUILD)

  # Initial cache info
  set( CACHE_CONTENTS "
SITE:STRING=${CTEST_SITE}
CMAKE_BUILD_TYPE:STRING=Release
CMAKE_GENERATOR:INTERNAL=${CTEST_CMAKE_GENERATOR}
BUILD_TESTING:BOOL=OFF
PRODUCT_REPO:STRING=${Albany_REPOSITORY_LOCATION}
" )

  ctest_empty_binary_directory( "${CTEST_BINARY_DIRECTORY}" )
  file(WRITE "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt" "${CACHE_CONTENTS}")

ENDIF()

IF (DOWNLOAD)

  # Get the Trilinos repo

  set(CTEST_CHECKOUT_COMMAND)

  if(NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/Trilinos")
    EXECUTE_PROCESS(COMMAND "${CTEST_GIT_COMMAND}" 
      clone ${Trilinos_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/Trilinos -b develop
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
    
    message(STATUS "out: ${_out}")
    message(STATUS "err: ${_err}")
    message(STATUS "res: ${HAD_ERROR}")
    if(HAD_ERROR)
      message(FATAL_ERROR "Cannot clone Trilinos repository!")
    endif()
  endif()

  set(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")

  # Get Albany

  if(NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/Albany")
    EXECUTE_PROCESS(COMMAND "${CTEST_GIT_COMMAND}" 
      clone ${Albany_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/Albany
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
    
    message(STATUS "out: ${_out}")
    message(STATUS "err: ${_err}")
    message(STATUS "res: ${HAD_ERROR}")
    if(HAD_ERROR)
      message(FATAL_ERROR "Cannot clone Albany repository!")
    endif()

  endif()

ENDIF()

ctest_start(${CTEST_TEST_TYPE})

# Send the project structure to CDash

IF(FALSE AND CTEST_DO_SUBMIT)
  CTEST_SUBMIT(FILES "${CTEST_SCRIPT_DIRECTORY}/Project.xml"
    RETURN_VALUE  HAD_ERROR
    )

  if(HAD_ERROR)
    message( "Cannot submit Albany Project.xml!")
  endif()
ENDIF()

IF(DOWNLOAD)

  # Update Trilinos
  SET_PROPERTY (GLOBAL PROPERTY SubProject Trilinos)
  SET_PROPERTY (GLOBAL PROPERTY Label Trilinos)

  ctest_update(SOURCE "${CTEST_SOURCE_DIRECTORY}/Trilinos" RETURN_VALUE count)
  message("Found ${count} changed files")

  IF(CTEST_DO_SUBMIT)
    CTEST_SUBMIT(PARTS Update
      RETURN_VALUE  HAD_ERROR
      )

    if(HAD_ERROR)
      message( "Cannot submit to cdash.")
    endif()
  ENDIF()

  # Update Albany branch
  SET_PROPERTY (GLOBAL PROPERTY SubProject Albany-meshDependsOnParameters)
  SET_PROPERTY (GLOBAL PROPERTY Label Albany-meshDependsOnParameters)

  set(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
  CTEST_UPDATE(SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany" RETURN_VALUE count)
  message("Found ${count} changed files")

  IF(CTEST_DO_SUBMIT)
    CTEST_SUBMIT(PARTS Update
      RETURN_VALUE  HAD_ERROR
      )

    if(HAD_ERROR)
      message( "Cannot submit to cdash.")
    endif()
  ENDIF()

ENDIF()


# Set the common Trilinos config options
SET(COMMON_CONFIGURE_OPTIONS
  "-Wno-dev"
  "-DCMAKE_BUILD_TYPE:STRING=NONE"
  #
  "-DTrilinos_ENABLE_ThyraTpetraAdapters:BOOL=ON"
  "-DTrilinos_ENABLE_Ifpack2:BOOL=ON"
  "-DTrilinos_ENABLE_Amesos2:BOOL=ON"
  "-DTrilinos_ENABLE_Zoltan2:BOOL=ON"
  "-DTrilinos_ENABLE_MueLu:BOOL=ON"
  #
  "-DZoltan_ENABLE_ULONG_IDS:BOOL=ON"
  "-DTeuchos_ENABLE_LONG_LONG_INT:BOOL=ON"
  "-DTeuchos_ENABLE_COMPLEX:BOOL=OFF"
  "-DZOLTAN_BUILD_ZFDRIVE:BOOL=OFF"
  #
  "-DTPL_Netcdf_PARALLEL:BOOL=ON"
  "-DSEACAS_ENABLE_SEACASSVDI:BOOL=OFF"
  "-DTrilinos_ENABLE_SEACASFastq:BOOL=OFF"
  "-DTrilinos_ENABLE_SEACASBlot:BOOL=OFF"
  "-DTrilinos_ENABLE_SEACASPLT:BOOL=OFF"
  "-DTPL_ENABLE_X11:BOOL=OFF"
  "-DTPL_ENABLE_Matio:BOOL=OFF"
  #
  "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF"
  "-DTrilinos_VERBOSE_CONFIGURE:BOOL=OFF"
  #
  "-DTPL_ENABLE_Boost:BOOL=ON"
  "-DTPL_ENABLE_BoostLib:BOOL=ON"
  "-DTPL_ENABLE_BoostAlbLib:BOOL=ON"
  "-DBoost_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DBoost_LIBRARY_DIRS:PATH=${PREFIX_DIR}/lib"
  "-DBoostLib_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DBoostLib_LIBRARY_DIRS:PATH=${PREFIX_DIR}/lib"
  "-DBoostAlbLib_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DBoostAlbLib_LIBRARY_DIRS:PATH=${PREFIX_DIR}/lib"
  #
  "-DTPL_ENABLE_Netcdf:STRING=ON"
  "-DTPL_Netcdf_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DTPL_Netcdf_LIBRARIES=${PREFIX_DIR}/lib/libnetcdf.a"
  "-DTPL_ENABLE_Pnetcdf:STRING=ON"
  "-DTPL_Pnetcdf_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DTPL_Pnetcdf_LIBRARIES=${PREFIX_DIR}/lib/libnetcdf.a"
  #
  "-DTPL_ENABLE_HDF5:STRING=ON"
  "-DTPL_HDF5_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DTPL_HDF5_LIBRARIES=${PREFIX_DIR}/lib/libhdf5_hl.a"
  "-DTrilinos_EXTRA_LINK_FLAGS:STRING='-L${PREFIX_DIR}/lib -lnetcdf -lpnetcdf -lhdf5_hl -lhdf5 -lz -lcurl -ldl'"
  #
  "-DTPL_ENABLE_Zlib:STRING=ON"
  "-DZlib_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DZlib_LIBRARY_DIRS:PATH=${PREFIX_DIR}/lib"
  #
  "-DTPL_ENABLE_ParMETIS:STRING=ON"
  "-DParMETIS_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DParMETIS_LIBRARY_DIRS:PATH=${PREFIX_DIR}/lib"
  #
  "-DTPL_ENABLE_SuperLU:STRING=ON"
  "-DSuperLU_INCLUDE_DIRS:PATH=${PREFIX_DIR}/SuperLU_4.3/include"
  "-DSuperLU_LIBRARY_DIRS:PATH=${PREFIX_DIR}/SuperLU_4.3/lib"
  #
  "-DTPL_BLAS_LIBRARIES:STRING='-L/usr/local/intel/11.1/069/mkl/lib/em64t -lmkl_intel_lp64 -lmkl_blas95_lp64 -lmkl_core -lmkl_sequential'"
  "-DTPL_LAPACK_LIBRARIES:STRING='-L/usr/local/intel/11.1/069/mkl/lib/em64t -lmkl_lapack95_lp64'"
  #
  "-DDART_TESTING_TIMEOUT:STRING=600"
  "-DTrilinos_ENABLE_ThreadPool:BOOL=ON"
  #
  "-DTrilinos_ENABLE_Kokkos:BOOL=ON"
  "-DTrilinos_ENABLE_KokkosCore:BOOL=ON"
  "-DPhalanx_KOKKOS_DEVICE_TYPE:STRING=SERIAL"
  "-DPhalanx_INDEX_SIZE_TYPE:STRING=INT"
  "-DPhalanx_SHOW_DEPRECATED_WARNINGS:BOOL=OFF"
  "-DKokkos_ENABLE_Serial:BOOL=ON"
  "-DKokkos_ENABLE_OpenMP:BOOL=OFF"
  "-DKokkos_ENABLE_Pthread:BOOL=OFF"
  #
  "-DTrilinos_ENABLE_TESTS:BOOL=OFF"
  "-DTrilinos_ENABLE_TriKota:BOOL=OFF"
  "-DTrilinos_ENABLE_EXPORT_MAKEFILES:BOOL=OFF"
  #  "-DTrilinos_ENABLE_EXPORT_MAKEFILES:BOOL=ON"
  "-DTrilinos_ASSERT_MISSING_PACKAGES:BOOL=OFF"
  #
  "-DTrilinos_ENABLE_ALL_PACKAGES:BOOL=OFF"
  "-DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF"
  "-DTrilinos_ENABLE_SECONDARY_TESTED_CODE:BOOL=ON"
  #
  "-DTrilinos_ENABLE_Teuchos:BOOL=ON"
  "-DTrilinos_ENABLE_Shards:BOOL=ON"
  "-DTrilinos_ENABLE_Sacado:BOOL=ON"
  "-DTrilinos_ENABLE_Epetra:BOOL=ON"
  "-DTrilinos_ENABLE_EpetraExt:BOOL=ON"
  "-DTrilinos_ENABLE_Ifpack:BOOL=ON"
  "-DTrilinos_ENABLE_AztecOO:BOOL=ON"
  "-DTrilinos_ENABLE_Amesos:BOOL=ON"
  "-DTrilinos_ENABLE_Anasazi:BOOL=ON"
  "-DTrilinos_ENABLE_Belos:BOOL=ON"
  "-DTrilinos_ENABLE_ML:BOOL=ON"
  "-DTrilinos_ENABLE_Phalanx:BOOL=ON"
  "-DTrilinos_ENABLE_Intrepid2:BOOL=ON"
  "-DTrilinos_ENABLE_MiniTensor:BOOL=ON"
  "-DTrilinos_ENABLE_NOX:BOOL=ON"
  "-DTrilinos_ENABLE_Stratimikos:BOOL=ON"
  "-DTrilinos_ENABLE_Thyra:BOOL=ON"
  "-DTrilinos_ENABLE_Rythmos:BOOL=ON"
  "-DTrilinos_ENABLE_OptiPack:BOOL=ON"
  "-DTrilinos_ENABLE_GlobiPack:BOOL=ON"
  "-DTrilinos_ENABLE_Stokhos:BOOL=ON"
  "-DTrilinos_ENABLE_Isorropia:BOOL=ON"
  "-DTrilinos_ENABLE_Piro:BOOL=ON"
  "-DTrilinos_ENABLE_Teko:BOOL=ON"
  "-DTrilinos_ENABLE_Zoltan:BOOL=ON"
  #
  "-DTrilinos_ENABLE_FEI:BOOL=OFF"
  #
  "-DPhalanx_ENABLE_TEUCHOS_TIME_MONITOR:BOOL=ON"
  "-DStokhos_ENABLE_TEUCHOS_TIME_MONITOR:BOOL=ON"
  "-DStratimikos_ENABLE_TEUCHOS_TIME_MONITOR:BOOL=ON"
  #
  "-DTrilinos_ENABLE_SEACAS:BOOL=ON"
  "-DTrilinos_ENABLE_Pamgen:BOOL=ON"
  "-DTrilinos_ENABLE_PyTrilinos:BOOL=OFF"
  #
  "-DTrilinos_ENABLE_STK:BOOL=ON"
  "-DTrilinos_ENABLE_STKClassic:BOOL=OFF"
  "-DTrilinos_ENABLE_STKUtil:BOOL=ON"
  "-DTrilinos_ENABLE_STKTopology:BOOL=ON"
  "-DTrilinos_ENABLE_STKMesh:BOOL=ON"
  "-DTrilinos_ENABLE_STKIO:BOOL=ON"
  "-DTrilinos_ENABLE_STKExp:BOOL=OFF"
  "-DTrilinos_ENABLE_STKSearch:BOOL=OFF"
  "-DTrilinos_ENABLE_STKSearchUtil:BOOL=OFF"
  "-DTrilinos_ENABLE_STKTransfer:BOOL=ON"
  "-DTrilinos_ENABLE_STKUnit_tests:BOOL=OFF"
  "-DTrilinos_ENABLE_STKDoc_tests:BOOL=OFF"
  )

IF(BUILD_TRILINOS)

  # Configure the Trilinos build
  SET_PROPERTY (GLOBAL PROPERTY SubProject Trilinos)
  SET_PROPERTY (GLOBAL PROPERTY Label Trilinos)

  SET(CONFIGURE_OPTIONS
    "-DTPL_ENABLE_MPI:BOOL=ON"
    "-DMPI_BASE_DIR:PATH=${PREFIX_DIR}/ompi-gcc"
    #
    "-DCMAKE_CXX_COMPILER:PATH=/users/ghansen/ompi-gcc/bin/mpicxx"
    "-DCMAKE_CXX_FLAGS:STRING='-O3 -march=native -w -DNDEBUG ${extra_cxx_flags}'"
    "-DCMAKE_C_COMPILER:PATH=/users/ghansen/ompi-gcc/bin/mpicc"
    "-DCMAKE_C_FLAGS:STRING='-O3 -march=native -w -DNDEBUG'"
    "-DCMAKE_Fortran_COMPILER:PATH=/users/ghansen/ompi-gcc/bin/mpifort"
    "-DCMAKE_Fortran_FLAGS:STRING='-O3 -march=native -w -DNDEBUG'"
    "-DCMAKE_INSTALL_PREFIX:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    ${COMMON_CONFIGURE_OPTIONS}
    )

  if(NOT EXISTS "${CTEST_BINARY_DIRECTORY}/TriBuild")
    FILE(MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/TriBuild)
  endif()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuild"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Trilinos"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    )

  if(HAD_ERROR)
    message(FATAL_ERROR "Cannot configure Trilinos build!")
  endif()

  IF(CTEST_DO_SUBMIT)
    CTEST_SUBMIT(PARTS Configure
      RETURN_VALUE  HAD_ERROR
      )

    if(HAD_ERROR)
      message( "Cannot submit Trilinos configure results!")
    endif()
  ENDIF()

  # Trilinos
  SET_PROPERTY (GLOBAL PROPERTY SubProject Trilinos)
  SET_PROPERTY (GLOBAL PROPERTY Label Trilinos)
  #SET(CTEST_BUILD_TARGET all)
  SET(CTEST_BUILD_TARGET install)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuild"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if(HAD_ERROR)
    message(FATAL_ERROR "Cannot build Trilinos!")
  endif()

  IF(CTEST_DO_SUBMIT)
    CTEST_SUBMIT(PARTS Build
      RETURN_VALUE  HAD_ERROR
      )

    if(HAD_ERROR)
      message( "Cannot submit Trilinos build results!")
    endif()

  ENDIF()

ENDIF(BUILD_TRILINOS)

IF(BUILD_TRILINOSCLANG)

  # Configure the Trilinos build
  SET_PROPERTY (GLOBAL PROPERTY SubProject TrilinosClang)
  SET_PROPERTY (GLOBAL PROPERTY Label TrilinosClang)

  SET(CONFIGURE_OPTIONS
    "-DTPL_ENABLE_MPI:BOOL=ON"
    "-DMPI_BASE_DIR:PATH=${PREFIX_DIR}/ompi-clang"
    #
    "-DTrilinos_ENABLE_CXX11:BOOL=ON"
    "-DCMAKE_CXX_COMPILER:PATH=/users/ghansen/ompi-clang/bin/mpicxx"
    "-DCMAKE_CXX_FLAGS:STRING='-O3 -w -DNDEBUG  ${extra_cxx_flags}'"
    "-DCMAKE_C_COMPILER:PATH=/users/ghansen/ompi-clang/bin/mpicc"
    "-DCMAKE_C_FLAGS:STRING='-O3 -w -DNDEBUG'"
    "-DCMAKE_Fortran_COMPILER:PATH=/users/ghansen/ompi-clang/bin/mpifort"
    "-DCMAKE_Fortran_FLAGS:STRING='-Os -w -DNDEBUG'"
    "-DMDS_ID_TYPE:STRING='long long int'"
    "-DCMAKE_INSTALL_PREFIX:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstallClang"
    "-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON"
    "-DTpetra_INST_FLOAT=OFF"
    "-DTpetra_INST_INT_INT=ON"
    "-DTpetra_INST_DOUBLE=ON"
    "-DTpetra_INST_COMPLEX_FLOAT=OFF"
    "-DTpetra_INST_COMPLEX_DOUBLE=OFF"
    "-DTpetra_INST_INT_LONG=OFF"
    "-DTpetra_INST_INT_UNSIGNED=OFF"
    "-DTpetra_INST_INT_LONG_LONG=ON"
    ${COMMON_CONFIGURE_OPTIONS}
    )

  if(NOT EXISTS "${CTEST_BINARY_DIRECTORY}/TriBuildClang")
    FILE(MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/TriBuildClang)
  endif()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuildClang"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Trilinos"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    )

  if(HAD_ERROR)
    message(FATAL_ERROR "Cannot configure TrilinosClang build!")
  endif()

  IF(CTEST_DO_SUBMIT)
    CTEST_SUBMIT(PARTS Configure
      RETURN_VALUE  HAD_ERROR
      )

    if(HAD_ERROR)
      message( "Cannot submit TrilinosClang configure results!")
    endif()
  ENDIF()

  #SET(CTEST_BUILD_TARGET all)
  SET(CTEST_BUILD_TARGET install)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  set(ENV{LD_LIBRARY_PATH} "/users/ghansen/ompi-clang/lib:${INITIAL_LD_LIBRARY_PATH}")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuildClang"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if(HAD_ERROR)
    message(FATAL_ERROR "Cannot build Trilinos with Clang!")
  endif()

  set(ENV{LD_LIBRARY_PATH} ${INITIAL_LD_LIBRARY_PATH})

  IF(CTEST_DO_SUBMIT)
    CTEST_SUBMIT(PARTS Build
      RETURN_VALUE  HAD_ERROR
      )

    if(HAD_ERROR)
      message( "Cannot submit TrilinoClang build results!")
    endif()

  ENDIF()

ENDIF(BUILD_TRILINOSCLANG)

IF (BUILD_ALB32)
  # Configure the Albany 32 Bit build 
  # Builds everything!
  SET_PROPERTY (GLOBAL PROPERTY SubProject Albany32Bit-DynRankViewRefactor)
  SET_PROPERTY (GLOBAL PROPERTY Label Albany32Bit-DynRankViewRefactor)

  SET(CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "-DENABLE_FELIX:BOOL=ON"
    "-D ENABLE_MESH_DEPENDS_ON_PARAMETERS:BOOL=ON"
    "-D ENABLE_DEMO_PDES:BOOL=ON"
    "-DENABLE_ALBANY_EPETRA_EXE:BOOL=ON"

    #  "-DENABLE_CHECK_FPE:BOOL=ON"
    )

  if(NOT EXISTS "${CTEST_BINARY_DIRECTORY}/Albany-meshDependsOnParameters")
    FILE(MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/Albany-meshDependsOnParameters)
  endif()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/Albany-meshDependsOnParameters"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    APPEND
    )

  if(HAD_ERROR)
    message(FATAL_ERROR "Cannot configure Albany build!")
  endif()

  IF(CTEST_DO_SUBMIT)
    CTEST_SUBMIT(PARTS Configure
      RETURN_VALUE  HAD_ERROR
      )

    if(HAD_ERROR)
      message( "Cannot submit Albany configure results!")
    endif()
  ENDIF()

  # Build Albany

  SET(CTEST_BUILD_TARGET all)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/Albany-meshDependsOnParameters"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if(HAD_ERROR)
    message(FATAL_ERROR "Cannot build Albany!")
  endif()

  IF(CTEST_DO_SUBMIT)
    CTEST_SUBMIT(PARTS Build
      RETURN_VALUE  HAD_ERROR
      )

    if(HAD_ERROR)
      message( "Cannot submit Albany build results!")
    endif()
  ENDIF()

  # Run Albany tests

  CTEST_TEST(
    BUILD "${CTEST_BINARY_DIRECTORY}/Albany-meshDependsOnParameters"
    #              PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
    #              INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
    #NUMBER_FAILED  TEST_NUM_FAILED
    )

  IF(CTEST_DO_SUBMIT)
    CTEST_SUBMIT(PARTS Test
      RETURN_VALUE  HAD_ERROR
      )

    if(HAD_ERROR)
      message( "Cannot submit Albany test results!")
    endif()
  ENDIF()

ENDIF(BUILD_ALB32)

# Configure the Albany build using GO = long
IF (BUILD_ALB64)
  SET_PROPERTY (GLOBAL PROPERTY SubProject Albany64Bit)
  SET_PROPERTY (GLOBAL PROPERTY Label Albany64Bit)

  SET(CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "-DENABLE_FELIX:BOOL=ON"
    "-D ENABLE_MESH_DEPENDS_ON_PARAMETERS:BOOL=ON"
    "-D ENABLE_DEMO_PDES:BOOL=ON"
    "-DENABLE_64BIT_INT:BOOL=ON"
    "-DENABLE_ALBANY_EPETRA_EXE:BOOL=ON"
    )

  if(NOT EXISTS "${CTEST_BINARY_DIRECTORY}/Albany64Bit")
    FILE(MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/Albany64Bit)
  endif()

  # The 64 bit build 

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/Albany64Bit"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    APPEND
    )

  # Read the CTestCustom.cmake file to turn off ignored tests

  #CTEST_READ_CUSTOM_FILES("${CTEST_BINARY_DIRECTORY}/AlbanyT64")

  if(HAD_ERROR)
    message("Cannot configure Albany 64 bit build!")
  endif()

  IF(CTEST_DO_SUBMIT)
    CTEST_SUBMIT(PARTS Configure
      RETURN_VALUE  HAD_ERROR
      )

    if(HAD_ERROR)
      message("Cannot submit Albany 64 bit configure results!")
    endif()
  ENDIF()

  # Build Albany 64 bit

  SET(CTEST_BUILD_TARGET all)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/Albany64Bit"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if(HAD_ERROR)
    message("Cannot build Albany 64 bit!")
  endif()

  IF(CTEST_DO_SUBMIT)
    CTEST_SUBMIT(PARTS Build
      RETURN_VALUE  HAD_ERROR
      )

    if(HAD_ERROR)
      message("Cannot submit Albany 64 bit build results!")
    endif()
  ENDIF()

  # Run Albany 64 bit tests

  CTEST_TEST(
    BUILD "${CTEST_BINARY_DIRECTORY}/Albany64Bit"
    #              PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
    #              INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
    #NUMBER_FAILED  TEST_NUM_FAILED
    )

  IF(CTEST_DO_SUBMIT)
    CTEST_SUBMIT(PARTS Test
      RETURN_VALUE  HAD_ERROR
      )

    if(HAD_ERROR)
      message("Cannot submit Albany 64 bit test results!")
    endif()
  ENDIF()
ENDIF(BUILD_ALB64)

# Configure the Albany Clang build using GO = long
IF (BUILD_ALB64CLANG)
  SET_PROPERTY (GLOBAL PROPERTY SubProject Albany64BitClang)
  SET_PROPERTY (GLOBAL PROPERTY Label Albany64BitClang)

  SET(CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstallClang"
    "-DENABLE_64BIT_INT:BOOL=ON"
    "-DENABLE_ALBANY_EPETRA_EXE:BOOL=ON"
    "-DENABLE_FELIX:BOOL=ON"
    "-D ENABLE_MESH_DEPENDS_ON_PARAMETERS:BOOL=ON"
    "-D ENABLE_DEMO_PDES:BOOL=ON"
    #  "-DENABLE_CHECK_FPE:BOOL=ON"
    )

  if(NOT EXISTS "${CTEST_BINARY_DIRECTORY}/Albany64BitClang")
    FILE(MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/Albany64BitClang)
  endif()

  # The 64 bit build 

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/Albany64BitClang"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    APPEND
    )

  # Read the CTestCustom.cmake file to turn off ignored tests

  #CTEST_READ_CUSTOM_FILES("${CTEST_BINARY_DIRECTORY}/AlbanyT64")

  if(HAD_ERROR)
    message(FATAL_ERROR "Cannot configure Albany 64 bit Clang build!")
  endif()

  IF(CTEST_DO_SUBMIT)
    CTEST_SUBMIT(PARTS Configure
      RETURN_VALUE  HAD_ERROR
      )

    if(HAD_ERROR)
      message( "Cannot submit Albany 64 bit Clang configure results!")
    endif()
  ENDIF()

  # Build Albany 64 bit

  SET(CTEST_BUILD_TARGET all)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/Albany64BitClang"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if(HAD_ERROR)
    message(FATAL_ERROR "Cannot build Albany 64 bit with Clang!")
  endif()

  IF(CTEST_DO_SUBMIT)
    CTEST_SUBMIT(PARTS Build
      RETURN_VALUE  HAD_ERROR
      )

    if(HAD_ERROR)
      message( "Cannot submit Albany 64 bit Clang build results!")
    endif()
  ENDIF()

  # Run Albany 64 bit tests

  CTEST_TEST(
    BUILD "${CTEST_BINARY_DIRECTORY}/Albany64BitClang"
    #              PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
    #              INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
    #NUMBER_FAILED  TEST_NUM_FAILED
    )

  IF(CTEST_DO_SUBMIT)
    CTEST_SUBMIT(PARTS Test
      RETURN_VALUE  HAD_ERROR
      )

    if(HAD_ERROR)
      message( "Cannot submit Albany 64 bit Clang test results!")
    endif()
  ENDIF()
ENDIF()

if (BUILD_ALBFUNCTOR)
  # ALBANY_KOKKOS_UNDER_DEVELOPMENT build

  set_property (GLOBAL PROPERTY SubProject AlbanyFunctorDev)
  set_property (GLOBAL PROPERTY Label AlbanyFunctorDev)

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "-DENABLE_FELIX:BOOL=ON"
    "-D ENABLE_MESH_DEPENDS_ON_PARAMETERS:BOOL=ON"
    "-D ENABLE_DEMO_PDES:BOOL=ON")
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/AlbanyFunctorDev")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/AlbanyFunctorDev)
  endif ()

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
    endif ()
  endif ()

  if (HAD_ERROR)
    message ("Cannot configure Albany build!")
    set (BUILD_ALBFUNCTOR FALSE)
  endif ()

  if (BUILD_ALBFUNCTOR)
    set (CTEST_BUILD_TARGET all)

    message ("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

    CTEST_BUILD (
      BUILD "${CTEST_BINARY_DIRECTORY}/AlbanyFunctorDev"
      RETURN_VALUE  HAD_ERROR
      NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
      APPEND)

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Build
        RETURN_VALUE  S_HAD_ERROR)

      if (S_HAD_ERROR)
        message ("Cannot submit Albany build results!")
        set (BUILD_ALBFUNCTOR FALSE)
      endif ()
    endif ()

    if (HAD_ERROR)
      message ("Cannot build Albany!")
      set (BUILD_ALBFUNCTOR FALSE)
    endif ()

    if (BUILD_LIBS_NUM_ERRORS GREATER 0)
      message ("Encountered build errors in Albany build.")
      set (BUILD_ALBFUNCTOR FALSE)
    endif ()
  endif ()

  if (BUILD_ALBFUNCTOR)
    set (CTEST_TEST_TIMEOUT 400)
    CTEST_TEST (
      BUILD "${CTEST_BINARY_DIRECTORY}/AlbanyFunctorDev"
      RETURN_VALUE HAD_ERROR)

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Test
        RETURN_VALUE S_HAD_ERROR)

      if (S_HAD_ERROR)
        message ("Cannot submit Albany test results!")
      endif ()
    endif ()
  endif ()
endif ()
