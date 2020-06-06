#cmake_minimum_required (VERSION 2.8)
set (CTEST_DO_SUBMIT ON)
set (CTEST_TEST_TYPE Nightly)

# What to build and test
set (CLEAN_BUILD FALSE)
set (DOWNLOAD_TRILINOS FALSE)
set (BUILD_TRILINOS TRUE)
set (DOWNLOAD_ALBANY FALSE) 

# Begin User inputs:
set (CTEST_SITE "mayer.sandia.gov" ) # generally the output of hostname
set (CTEST_DASHBOARD_ROOT "$ENV{TEST_DIRECTORY}" ) # writable path
set (CTEST_SCRIPT_DIRECTORY "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set (CTEST_CMAKE_GENERATOR "Unix Makefiles" ) # What is your compilation apps ?
set (CTEST_CONFIGURATION  Release) # What type of build do you want ?

set (INITIAL_LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})

set (CTEST_PROJECT_NAME "Albany" )
set (CTEST_SOURCE_NAME repos)
set (CTEST_BUILD_CONFIGURATION Release)
set (CTEST_BUILD_NAME "mayer-${CTEST_BUILD_CONFIGURATION}-Trilinos")
set (CTEST_BINARY_NAME build)


set (CTEST_SOURCE_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_SOURCE_NAME}")
set (CTEST_BINARY_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_BINARY_NAME}")

if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}")
  file (MAKE_DIRECTORY "${CTEST_SOURCE_DIRECTORY}")
endif ()
if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}")
  file (MAKE_DIRECTORY "${CTEST_BINARY_DIRECTORY}")
endif ()

configure_file (${CTEST_SCRIPT_DIRECTORY}/CTestConfig.cmake
  ${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake COPYONLY)

set (CTEST_NIGHTLY_START_TIME "01:00:00 UTC")
set (CTEST_CMAKE_COMMAND "cmake")
set (CTEST_COMMAND "ctest -D ${CTEST_TEST_TYPE}")
set (CTEST_FLAGS "-j8")
SET (CTEST_BUILD_FLAGS "-j8")


find_program (CTEST_GIT_COMMAND NAMES git)

set (Albany_REPOSITORY_LOCATION git@github.com:SNLComputation/Albany.git)
set (Trilinos_REPOSITORY_LOCATION git@github.com:trilinos/Trilinos.git)
set (HDF5_DIR $ENV{HDF5_DIR})
set (NETCDF_DIR $ENV{NETCDF_DIR}) 
set (PNETCDF_DIR $ENV{PNETCDF_DIR}) 
set (BOOST_DIR $ENV{BOOST_DIR}) 
set (BLAS_DIR $ENV{OPENBLAS_DIR}) 
set (LAPACK_DIR $ENV{OPENBLAS_DIR}) 
set (ARMPL_DIR $ENV{ARMPL_DIR})
set (ZLIB_DIR $ENV{ZLIB_DIR})  
set (MPI_DIR $ENV{MPI_DIR})  

if (CLEAN_BUILD)
  # Initial cache info
  set (CACHE_CONTENTS "
  SITE:STRING=${CTEST_SITE}
  CMAKE_TYPE:STRING=Release
  CMAKE_GENERATOR:INTERNAL=${CTEST_CMAKE_GENERATOR}
  TESTING:BOOL=OFF
  PRODUCT_REPO:STRING=${Albany_REPOSITORY_LOCATION}
  " )

  ctest_empty_binary_directory( "${CTEST_BINARY_DIRECTORY}" )
  file(WRITE "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt" "${CACHE_CONTENTS}")
endif ()

if (DOWNLOAD_TRILINOS)

  set (CTEST_CHECKOUT_COMMAND)
 
  #
  # Get Trilinos
  #
  
  if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/Trilinos")
    execute_process (COMMAND "${CTEST_GIT_COMMAND}" 
      clone ${Trilinos_REPOSITORY_LOCATION} -b develop ${CTEST_SOURCE_DIRECTORY}/Trilinos
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
    message(STATUS "out: ${_out}")
    message(STATUS "err: ${_err}")
    message(STATUS "res: ${HAD_ERROR}")
    if (HAD_ERROR)
      message(FATAL_ERROR "Cannot clone Trilinos repository!")
    endif ()
  endif ()

  # Pull the repo
  execute_process (COMMAND "${CTEST_GIT_COMMAND}" pull
      WORKING_DIRECTORY ${CTEST_SOURCE_DIRECTORY}/Trilinos
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
  message(STATUS "Output of Trilinos pull: ${_out}")
  message(STATUS "Text sent to standard error stream: ${_err}")
  message(STATUS "command result status: ${HAD_ERROR}")
  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot pull Trilinos!")
  endif ()

endif()


ctest_start(${CTEST_TEST_TYPE})

# 
# Set the common Trilinos config options & build Trilinos
# 

if (BUILD_TRILINOS) 
  message ("ctest state: BUILD_TRILINOS")
  #
  # Configure the Trilinos/SCOREC build
  #
  set (CONFIGURE_OPTIONS
    "-DCMAKE_INSTALL_PREFIX:PATH=$ENV{jenkins_trilinos_install_dir}"
    "-DKokkos_ENABLE_PTHREAD=OFF"
    "-DKokkos_ENABLE_SERIAL:BOOL=ON"
    "-DKokkos_ENABLE_OPENMP:BOOL=OFF"
    #
    "-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON"
    "-DTrilinos_ENABLE_DEBUG:BOOL=OFF"
    "-DCMAKE_BUILD_TYPE:STRING=RELEASE"
    #
    "-DCMAKE_C_COMPILER=mpicc"
    "-DCMAKE_CXX_COMPILER=mpicxx"
    "-DCMAKE_Fortran_COMPILER=mpif90"
    "-DCMAKE_CXX_FLAGS:STRING='-Wno-inconsistent-missing-override -Wno-deprecated-declarations'"
    "-DTPL_DLlib_LIBRARIES='dl'"
    "-DTrilinos_ENABLE_INSTALL_CMAKE_CONFIG_FILES:BOOL=ON"
    "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF"
    "-DTrilinos_ENABLE_CHECKED_STL:BOOL=OFF"
    "-DTrilinos_ENABLE_ALL_PACKAGES:BOOL=OFF"
    "-DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF"
    "-DDART_TESTING_TIMEOUT:STRING=600"
    "-DTrilinos_ENABLE_CXX11=ON"
    #
    "-DTPL_ENABLE_MPI=ON"
    "-DTPL_ENABLE_BinUtils=OFF"
    "-DTPL_ENABLE_SuperLU=OFF"
    "-DTPL_ENABLE_BLAS:BOOL=ON"
    "-DBLAS_INCLUDE_DIRS:PATH=${ARMPL_DIR}/include"
    "-DTPL_BLAS_LIBRARIES:STRING=${ARMPL_DIR}/lib/libarmpl_lp64.so"
    "-DTPL_ENABLE_LAPACK:BOOL=ON"
    "-DLAPACK_INCLUDE_DIRS:PATH=${ARMPL_DIR}/include"
    "-DTPL_LAPACK_LIBRARIES:STRING=${ARMPL_DIR}/lib/libarmpl_lp64.so"
    "-DTPL_ENABLE_Boost=ON"
    "-DBoost_INCLUDE_DIRS:PATH=${BOOST_DIR}/include"
    "-DBoost_LIBRARY_DIRS:PATH=${BOOST_DIR}/lib"
    "-DTPL_ENABLE_BoostLib=ON"
    "-DBoostLib_INCLUDE_DIRS:PATH=${BOOST_DIR}/include"
    "-DBoostLib_LIBRARY_DIRS:PATH=${BOOST_DIR}/lib"
    "-DTPL_ENABLE_Netcdf=ON"
    "-DNetcdf_INCLUDE_DIRS:PATH=${NETCDF_DIR}/include"
    "-DNetcdf_LIBRARY_DIRS:PATH=${NETCDF_DIR}/lib"
    "-DTPL_Netcdf_PARALLEL:BOOL=ON"
    "-DTPL_ENABLE_HDF5=ON"
    "-DHDF5_INCLUDE_DIRS:PATH=${HDF5_DIR}/include"
    "-DHDF5_LIBRARY_DIRS:PATH=${HDF5_DIR}/lib"
    "-DTPL_ENABLE_Zlib:BOOL=ON"
    "-DZlib_INCLUDE_DIRS:PATH=${ZLIB_DIR}/include"
    "-DZlib_LIBRARY_DIRS:PATH=${ZLIB_DIR}/lib"
    "-DTrilinos_EXTRA_LINK_FLAGS:STRING='-lnetcdf -lpnetcdf -lhdf5_hl -lhdf5 -lz'"
    #
    "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF"
    "-DTrilinos_VERBOSE_CONFIGURE:BOOL=OFF"
    "-DTrilinos_ENABLE_ALL_PACKAGES:BOOL=OFF"
    "-DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF"
    "-DTrilinos_ENABLE_SECONDARY_TESTED_CODE:BOOL=ON"
    "-DTrilinos_ENABLE_EXPORT_MAKEFILES:BOOL=OFF"
    "-DTrilinos_ASSERT_MISSING_PACKAGES:BOOL=OFF"
    "-DTrilinos_WARNINGS_AS_ERRORS_FLAGS:STRING=''"
    "-DTeuchos_ENABLE_COMPLEX:BOOL=OFF"
    #
    "-DTPL_ENABLE_Matio=OFF"
    "-DTrilinos_ENABLE_TESTS:BOOL=OFF"
    "-DTrilinos_ENABLE_MiniTensor:BOOL=ON"
    "-DTrilinos_ENABLE_TriKota:BOOL=OFF"
    "-DTrilinos_ENABLE_Teuchos:BOOL=ON"
    "-DTrilinos_ENABLE_Shards:BOOL=ON"
    "-DTrilinos_ENABLE_Sacado:BOOL=ON"
    "-DTrilinos_ENABLE_Epetra:BOOL=ON"
    "-DTrilinos_ENABLE_Tempus:BOOL=ON"
    "-DTrilinos_ENABLE_EpetraExt:BOOL=ON"
    "-DTrilinos_ENABLE_Ifpack:BOOL=ON"
    "-DTrilinos_ENABLE_AztecOO:BOOL=ON"
    "-DTrilinos_ENABLE_Amesos:BOOL=ON"
    "-DTrilinos_ENABLE_Anasazi:BOOL=ON"
    "-DTrilinos_ENABLE_Belos:BOOL=ON"
    "-DTrilinos_ENABLE_ML:BOOL=ON"
    "-DTrilinos_ENABLE_Phalanx:BOOL=ON"
    "-DPhalanx_ENABLE_TESTS:BOOL=OFF"
    "-DPhalanx_ENABLE_EXAMPLES:BOOL=OFF"
    "-DTrilinos_ENABLE_Intrepid:BOOL=ON"
    "-DTrilinos_ENABLE_Intrepid2:BOOL=ON"
    "-DIntrepid2_ENABLE_TESTS:BOOL=OFF"
    "-DIntrepid2_ENABLE_EXAMPLES:BOOL=OFF"
    "-DTrilinos_ENABLE_ROL:BOOL=ON"
    "-DTrilinos_ENABLE_NOX:BOOL=ON"
    "-DTrilinos_ENABLE_Stratimikos:BOOL=ON"
    "-DTrilinos_ENABLE_Thyra:BOOL=ON"
    "-DTrilinos_ENABLE_Rythmos:BOOL=ON"
    "-DTrilinos_ENABLE_Piro:BOOL=ON"
    "-DTrilinos_ENABLE_Pamgen:BOOL=ON"
    "-DTrilinos_ENABLE_PanzerExprEval:BOOL=ON"
    "-DTrilinos_ENABLE_Isorropia:BOOL=ON"
    "-DTrilinos_ENABLE_Teko:BOOL=ON"
    "-DTrilinos_ENABLE_PyTrilinos:BOOL=OFF"
    #
    "-DTrilinos_ENABLE_STK:BOOL=ON"
    "-DTrilinos_ENABLE_STKExp:BOOL=OFF"
    "-DTrilinos_ENABLE_STKDoc_tests:BOOL=OFF"
    "-DTrilinos_ENABLE_STKIO:BOOL=ON"
    "-DTrilinos_ENABLE_STKMesh:BOOL=ON"
    "-DTrilinos_ENABLE_STKSearch:BOOL=ON"
    "-DTrilinos_ENABLE_STKSearchUtil:BOOL=OFF"
    "-DTrilinos_ENABLE_STKTopology:BOOL=ON"
    "-DTrilinos_ENABLE_STKTransfer:BOOL=ON"
    "-DTrilinos_ENABLE_STKUnit_tests:BOOL=OFF"
    "-DTrilinos_ENABLE_STKUtil:BOOL=ON"
    "-DTrilinos_ENABLE_STKExprEval:BOOL=ON"
    #
    "-DTrilinos_ENABLE_SEACAS:BOOL=ON"
    "-DTrilinos_ENABLE_SEACASIoss:BOOL=ON"
    "-DTrilinos_ENABLE_SEACASExodus:BOOL=ON"
    "-DSEACAS_ENABLE_SEACASSVDI:BOOL=OFF"
    "-DTrilinos_ENABLE_SEACASFastq:BOOL=OFF"
    "-DTrilinos_ENABLE_SEACASBlot:BOOL=OFF"
    "-DTrilinos_ENABLE_SEACASPLT:BOOL=OFF"
    "-DTPL_ENABLE_X11:BOOL=OFF"
    "-DTrilinos_ENABLE_Tpetra:BOOL=ON"
    "-DTrilinos_ENABLE_Kokkos:BOOL=ON"
    "-DTrilinos_ENABLE_Ifpack2:BOOL=ON"
    "-DTrilinos_ENABLE_Amesos2:BOOL=ON"
    "-DTrilinos_ENABLE_Zoltan2:BOOL=ON"
    "-DTrilinos_ENABLE_Zoltan:BOOL=ON"
    "-DZoltan_ENABLE_ULONG_IDS:BOOL=ON"
    "-DZOLTAN_BUILD_ZFDRIVE:BOOL=OFF"
    "-DTrilinos_ENABLE_FEI:BOOL=OFF"
    "-DPhalanx_ENABLE_TEUCHOS_TIME_MONITOR:BOOL=ON"
    "-DStratimikos_ENABLE_TEUCHOS_TIME_MONITOR:BOOL=ON"
    "-DTrilinos_ENABLE_MueLu:BOOL=ON"
    "-DAmesos2_ENABLE_KLU2:BOOL=ON"
    "-DAnasazi_ENABLE_RBGen:BOOL=OFF"
    #
    "-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON"
    "-DTpetra_INST_INT_LONG_LONG:BOOL=ON"
    "-DTpetra_INST_INT_INT:BOOL=OFF"
    "-DTpetra_INST_DOUBLE:BOOL=ON"
    "-DTpetra_INST_FLOAT:BOOL=OFF"
    "-DTpetra_INST_COMPLEX_FLOAT:BOOL=OFF"
    "-DTpetra_INST_COMPLEX_DOUBLE:BOOL=OFF"
    "-DTpetra_INST_INT_LONG:BOOL=OFF"
    "-DTpetra_INST_INT_UNSIGNED:BOOL=OFF"
    #
    "-DTrilinos_ENABLE_Kokkos:BOOL=ON"
    "-DTrilinos_ENABLE_KokkosCore:BOOL=ON"
    "-DPhalanx_KOKKOS_DEVICE_TYPE:STRING='SERIAL'"
    "-DPhalanx_INDEX_SIZE_TYPE:STRING='INT'"
    "-DPhalanx_SHOW_DEPRECATED_WARNINGS:BOOL=OFF"
    "-DTrilinos_ENABLE_OpenMP:BOOL=OFF"
    "-DTPL_ENABLE_HWLOC:STRING=OFF"
    "-DKokkos_ARCH_ARMV8_THUNDERX2=ON"
    #
    "-DTrilinos_ENABLE_Panzer:BOOL=OFF"
    "-DMPI_EXEC=${MPI_DIR}/bin/mpirun"
    "-DMPI_EXEC_MAX_NUMPROCS:STRING='4'"
    "-DMPI_EXEC_NUMPROCS_FLAG:STRING='-np'"
    "-DTpetra_ENABLE_DEPRECATED_CODE:BOOL=OFF"
    "-DXpetra_ENABLE_DEPRECATED_CODE:BOOL=OFF"
  )
    
  #"-DMPI_EXEC_POST_NUMPROCS_FLAGS:STRING='-bind-to;numa;-map-by;numa;'"

  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/TriBuild")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/TriBuild)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuild"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Trilinos"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit Trilinos configure results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message ("Cannot configure Trilinos build!")
  endif ()

  #
  # Build the rest of Trilinos and install everything
  #

  #set (CTEST_BUILD_TARGET all)
  set (CTEST_BUILD_TARGET install)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuild"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Build
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit Trilinos build results!")
    endif ()

  endif ()

  if (HAD_ERROR)
    message ("Cannot build Trilinos!")
  endif ()

  if (BUILD_LIBS_NUM_ERRORS GREATER 0)
    message ("Encountered build errors in Trilinos build. Exiting!")
  endif ()
  
  #
  # Run Trilinos tests
  #

  set (CTEST_TEST_TIMEOUT 500)
  CTEST_TEST (
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuild"
    RETURN_VALUE HAD_ERROR)

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Test RETURN_VALUE S_HAD_ERROR)

    if (S_HAD_ERROR)
      message ("Cannot submit Trilinos test results!")
    endif ()
  endif ()


endif()
