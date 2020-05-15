
#cmake_minimum_required (VERSION 2.8)
set (CTEST_DO_SUBMIT ON)
set (CTEST_TEST_TYPE Nightly)

# What to build and test
set (CLEAN_BUILD TRUE)
set (DOWNLOAD_TRILINOS TRUE)
set (BUILD_TRILINOS_SERIAL TRUE)
set (BUILD_TRILINOS_OPENMP FALSE)
set (DOWNLOAD_ALBANY FALSE) 
set (BUILD_ALBANY_SERIAL FALSE) 
set (BUILD_ALBANY_OPENMP FALSE) 


# Begin User inputs:
set (CTEST_SITE "blake.sandia.gov" ) # generally the output of hostname
set (CTEST_DASHBOARD_ROOT "$ENV{TEST_DIRECTORY}" ) # writable path
set (CTEST_SCRIPT_DIRECTORY "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set (CTEST_CMAKE_GENERATOR "Unix Makefiles" ) # What is your compilation apps ?
set (CTEST_CONFIGURATION  Release) # What type of build do you want ?

set (INITIAL_LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})

set (CTEST_PROJECT_NAME "Albany" )
set (CTEST_SOURCE_NAME repos)
set (CTEST_BUILD_NAME "blake-serial-Trilinos")
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
set (MPI_PATH $ENV{MPI_ROOT})  
set (MKL_PATH $ENV{MKL_ROOT})  
set (BOOST_PATH $ENV{BOOST_ROOT}) 
set (NETCDF_PATH $ENV{NETCDF_ROOT}) 
set (HDF5_PATH $ENV{HDF5_ROOT})
set (ZLIB_PATH $ENV{ZLIB_ROOT})  
set (YAMLCPP_PATH $ENV{YAMLCPP_ROOT})

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
if (BUILD_TRILINOS_SERIAL) 
  message ("ctest state: BUILD_TRILINOS_SERIAL")
  #
  # Configure the Trilinos/SCOREC build
  #
  set (CONFIGURE_OPTIONS
      "-DCMAKE_INSTALL_PREFIX:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosSerialInstall"
      "-DCMAKE_BUILD_TYPE:STRING=RELEASE"
      "-DCMAKE_CXX_COMPILER=mpicxx" 
      "-DCMAKE_C_COMPILER=mpicc"
      "-DCMAKE_Fortran_COMPILER=mpif90" 
      "-DCMAKE_C_FLAGS:STRING='-mkl'"
      "-DCMAKE_CXX_FLAGS:STRING='-mkl'" 
      "-DCMAKE_Fortran_FLAGS:STRING='-mkl'" 
      "-DCMAKE_EXE_LINKER_FLAGS='-mkl'"
      "-DTPL_ENABLE_MPI:BOOL=ON"
      "-DTPL_MPI_INCLUDE_DIRS:STRING=${MPI_PATH}/include"
      "-DTPL_MPI_LIBRARY_DIRS:STRING=${MPI_PATH}/lib"
      "-DTPL_ENABLE_BLAS:BOOL=ON"
      "-DTPL_BLAS_LIBRARIES:STRING=${MKL_PATH}/lib/intel64/libmkl_core.a"
      "-DTPL_ENABLE_LAPACK:BOOL=ON"
      "-DTPL_LAPACK_LIBRARIES:STRING=${MKL_PATH}/lib/intel64/libmkl_core.a"
      "-DTPL_ENABLE_Boost:BOOL=ON"
      "-DBoost_INCLUDE_DIRS:PATH=${BOOST_PATH}/include"
      "-DBoost_LIBRARY_DIRS:PATH=${BOOST_PATH}/lib"
      "-DTPL_ENABLE_BoostLib:BOOL=ON"
      "-DBoostLib_INCLUDE_DIRS:PATH=${BOOST_PATH}/include" 
      "-DBoostLib_LIBRARY_DIRS:PATH=${BOOST_PATH}/lib" 
      "-DTPL_ENABLE_Netcdf:BOOL=ON"
      "-DNetcdf_INCLUDE_DIRS:PATH=${NETCDF_PATH}/include"
      "-DNetcdf_LIBRARY_DIRS:PATH=${NETCDF_PATH}/lib"
      "-DTPL_Netcdf_PARALLEL:BOOL=ON"
      "-DTPL_ENABLE_HDF5:STRING=ON"
      "-DHDF5_INCLUDE_DIRS:PATH=${HDF5_PATH}/include"
      "-DHDF5_LIBRARY_DIRS:PATH=${HDF5_PATH}/lib"
      "-DTPL_ENABLE_Zlib:BOOL=ON"
      "-DZlib_INCLUDE_DIRS:PATH=${ZLIB_PATH}/include"
      "-DZlib_LIBRARY_DIRS:PATH=${ZLIB_PATH}/lib"
      "-DTPL_ENABLE_yaml-cpp:BOOL=ON"
      "-Dyaml-cpp_INCLUDE_DIRS:PATH=${YAMLCPP_PATH}/include"
      "-Dyaml-cpp_LIBRARY_DIRS:PATH=${YAMLCPP_PATH}/lib"
      "-DTrilinos_ENABLE_CXX11:BOOL=ON"
      "-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON"
      "-DTrilinos_VERBOSE_CONFIGURE:BOOL=OFF"
      "-DTPL_ENABLE_Matio=OFF"
      "-DTPL_ENABLE_SimMesh=OFF"
      "-DTPL_ENABLE_SimModel=OFF"
      "-DTPL_ENABLE_X11=OFF"
      "-DKOKKOS_ARCH:STRING='SKX'"
      "-DKokkos_ENABLE_CUDA_LAMBDA:BOOL=OFF"
      "-DKokkos_ENABLE_CUDA_UVM:BOOL=OFF"
      "-DKokkos_ENABLE_EXAMPLES:BOOL=OFF"
      "-DKokkos_ENABLE_OPENMP:BOOL=OFF"
      "-DKokkos_ENABLE_PTHREAD:BOOL=OFF"
      "-DKokkos_ENABLE_SERIAL:BOOL=ON"
      "-DKokkos_ENABLE_TESTS:BOOL=OFF"
      "-DTPL_ENABLE_Thrust:BOOL=OFF"
      "-DTPL_ENABLE_CUDA:BOOL=OFF"
      "-DTPL_ENABLE_CUSPARSE:BOOL=OFF"
      "-DAmesos2_ENABLE_KLU2:BOOL=ON"
      "-DAnasazi_ENABLE_RBGen:BOOL=OFF"
      "-DEpetraExt_USING_HDF5:BOOL=OFF"
      "-DIntrepid_ENABLE_TESTS:BOOL=OFF"
      "-DIntrepid2_ENABLE_TESTS:BOOL=OFF"
      "-DPhalanx_INDEX_SIZE_TYPE:STRING='INT'"
      "-DPhalanx_KOKKOS_DEVICE_TYPE:STRING='SERIAL'"
      "-DPhalanx_SHOW_DEPRECATED_WARNINGS:BOOL=OFF"
      "-DSacado_ENABLE_COMPLEX:BOOL=OFF"
      "-DTeuchos_ENABLE_COMPLEX:BOOL=OFF"
      "-DTpetra_ENABLE_Kokkos_Refactor:BOOL=ON"
      "-DTpetra_ASSUME_CUDA_AWARE_MPI:BOOL=OFF"
      "-DTpetra_INST_INT_LONG_LONG:BOOL=ON"
      "-DTpetra_INST_INT_INT:BOOL=OFF"
      "-DZoltan2_ENABLE_Experimental:BOOL=ON"
      "-DZoltan_ENABLE_ULONG_IDS:BOOL=ON"
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
      "-DTrilinos_ENABLE_Intrepid:BOOL=ON"
      "-DTrilinos_ENABLE_Intrepid2:BOOL=ON"
      "-DTrilinos_ENABLE_Kokkos:BOOL=ON"
      "-DTrilinos_ENABLE_KokkosAlgorithms:BOOL=ON"
      "-DTrilinos_ENABLE_KokkosContainers:BOOL=ON"
      "-DTrilinos_ENABLE_KokkosCore:BOOL=ON"
      "-DTrilinos_ENABLE_ML:BOOL=ON"
      "-DTrilinos_ENABLE_MiniTensor:BOOL=ON"
      "-DTrilinos_ENABLE_OpenMP:BOOL=OFF"
      "-DTrilinos_ENABLE_MueLu:BOOL=ON"
      "-DTrilinos_ENABLE_NOX:BOOL=ON"
      "-DTrilinos_ENABLE_Pamgen:BOOL=ON"
      "-DTrilinos_ENABLE_PanzerExprEval:BOOL=ON"
      "-DTrilinos_ENABLE_Phalanx:BOOL=ON"
      "-DTrilinos_ENABLE_Piro:BOOL=ON"
      "-DTrilinos_ENABLE_ROL:BOOL=ON"
      "-DTrilinos_ENABLE_Rythmos:BOOL=ON"
      "-DTrilinos_ENABLE_SEACAS:BOOL=ON"
      "-DTrilinos_ENABLE_SEACASAprepro_lib:BOOL=ON"
      "-DTrilinos_ENABLE_SEACASExodus:BOOL=ON"
      "-DTrilinos_ENABLE_SEACASIoss:BOOL=ON"
      "-DTrilinos_ENABLE_STKDoc_tests:BOOL=OFF"
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
      "-DTrilinos_ENABLE_Thyra:BOOL=ON"
      "-DTrilinos_ENABLE_ThyraTpetraAdapters:BOOL=ON"
      "-DTrilinos_ENABLE_Tpetra:BOOL=ON"
      "-DTrilinos_ENABLE_TrilinosCouplings:BOOL=ON"
      "-DTrilinos_ENABLE_TriKota:BOOL=OFF"
      "-DTrilinos_ENABLE_Zoltan2:BOOL=ON"
      "-DTrilinos_ENABLE_Zoltan:BOOL=ON"
      "-DPhalanx_ALLOW_MULTIPLE_EVALUATORS_FOR_SAME_FIELD:BOOL=ON"
      "-DTpetra_ENABLE_DEPRECATED_CODE:BOOL=OFF"
      "-DXpetra_ENABLE_DEPRECATED_CODE:BOOL=OFF"

  )

  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/TriBuildSerial")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/TriBuildSerial)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuildSerial"
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
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuildSerial"
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
  set (CTEST_TEST_TIMEOUT 600)

  CTEST_TEST(
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuildSerial"
    RETURN_VALUE  HAD_ERROR
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Test
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit Trilinos test results!")
    endif ()
  endif ()

endif()
