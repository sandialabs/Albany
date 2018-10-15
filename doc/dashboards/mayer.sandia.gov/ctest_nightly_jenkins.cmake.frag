
# Begin User inputs:
set (CTEST_SITE "mayer.sandia.gov" ) # generally the output of hostname
set (CTEST_DASHBOARD_ROOT "$ENV{TEST_DIRECTORY}" ) # writable path
set (CTEST_SCRIPT_DIRECTORY "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set (CTEST_CMAKE_GENERATOR "Unix Makefiles" ) # What is your compilation apps ?
set (CTEST_CONFIGURATION  Release) # What type of build do you want ?

set (INITIAL_LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})

set (CTEST_PROJECT_NAME "Albany" )
set (CTEST_SOURCE_NAME repos)
set (CTEST_NAME "linux-gcc-${CTEST_BUILD_CONFIGURATION}")
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

set (CTEST_NIGHTLY_START_TIME "00:00:00 UTC")
set (CTEST_CMAKE_COMMAND "cmake")
set (CTEST_COMMAND "ctest -D ${CTEST_TEST_TYPE}")
set (CTEST_FLAGS "-j8")
SET (CTEST_BUILD_FLAGS "-j8")

#set (CTEST_DROP_METHOD "http")

#if (CTEST_DROP_METHOD STREQUAL "http")
#  set (CTEST_DROP_SITE "cdash.sandia.gov")
#  set (CTEST_PROJECT_NAME "Albany")
#  set (CTEST_DROP_LOCATION "/CDash-2-3-0/submit.php?project=Albany")
#  set (CTEST_TRIGGER_SITE "")
#  set (CTEST_DROP_SITE_CDASH TRUE)
#endif ()

find_program (CTEST_GIT_COMMAND NAMES git)

set (Albany_REPOSITORY_LOCATION git@github.com:gahansen/Albany.git)
set (Trilinos_REPOSITORY_LOCATION git@github.com:trilinos/Trilinos.git)
set (HDF5_PATH $ENV{HDF5_ROOT})
set (NETCDF_PATH $ENV{NETCDF_ROOT}) 
set (PNETCDF_PATH $ENV{PNETCDF_ROOT}) 
set (BOOST_PATH $ENV{BOOST_ROOT}) 
set (BLAS_PATH $ENV{OPENBLAS_ROOT}) 
set (LAPACK_PATH $ENV{OPENBLAS_ROOT}) 
set (ZLIB_PATH $ENV{ZLIB_DIR})  

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


if (DOWNLOAD_ALBANY)

  set (CTEST_CHECKOUT_COMMAND)
  set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
  
  #
  # Get Albany
  #

  if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/Albany")
    execute_process (COMMAND "${CTEST_GIT_COMMAND}" 
      clone ${Albany_REPOSITORY_LOCATION} -b master ${CTEST_SOURCE_DIRECTORY}/Albany
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
    
    message(STATUS "out: ${_out}")
    message(STATUS "err: ${_err}")
    message(STATUS "res: ${HAD_ERROR}")
    if (HAD_ERROR)
      message(FATAL_ERROR "Cannot clone Albany repository!")
    endif ()
  endif ()

  set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
  
  # Pull the repo
  execute_process (COMMAND "${CTEST_GIT_COMMAND}" pull
      WORKING_DIRECTORY ${CTEST_SOURCE_DIRECTORY}/Albany
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
  message(STATUS "Output of Trilinos pull: ${_out}")
  message(STATUS "Text sent to standard error stream: ${_err}")
  message(STATUS "command result status: ${HAD_ERROR}")
  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot pull Albany!")
  endif ()

endif ()


ctest_start(${CTEST_TEST_TYPE})

#
# Send the project structure to CDash
#

#if (CTEST_DO_SUBMIT)
#  ctest_submit (FILES "${CTEST_SCRIPT_DIRECTORY}/Project.xml"
#    RETURN_VALUE  HAD_ERROR
#    )

#  if (HAD_ERROR)
#    message(FATAL_ERROR "Cannot submit Albany Project.xml!")
#  endif ()
#endif ()

# 
# Set the common Trilinos config options & build Trilinos
# 

if (BUILD_TRILINOS) 
  message ("ctest state: BUILD_TRILINOS")
  #
  # Configure the Trilinos/SCOREC build
  #
  set_property (GLOBAL PROPERTY SubProject IKTMayerARMTrilinos)
  set_property (GLOBAL PROPERTY Label IKTMayerARMTrilinos)

  set (CONFIGURE_OPTIONS
    "-DTrilinos_ENABLE_OpenMP=ON"
    "-DKokkos_ENABLE_Pthread=OFF"
    "-DTeuchos_ENABLE_COMPLEX=ON"
    #
    "-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON"
    "-DTrilinos_ENABLE_DEBUG:BOOL=OFF"
    "-DTPL_FIND_SHARED_LIBS:BOOL=OFF"
    #
    "-DCMAKE_INSTALL_PREFIX:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "-DCMAKE_C_COMPILER=mpicc"
    "-DCMAKE_CXX_COMPILER=mpicxx"
    "-DCMAKE_Fortran_COMPILER=mpif90"
    "-DTPL_DLlib_LIBRARIES='dl'"
    "-DTrilinos_ENABLE_INSTALL_CMAKE_CONFIG_FILES:BOOL=ON"
    "-DCMAKE_BUILD_TYPE:STRING=RELEASE"
    "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF"
    "-DTrilinos_ENABLE_CHECKED_STL:BOOL=OFF"
    "-DTrilinos_ENABLE_ALL_PACKAGES:BOOL=OFF"
    "-DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF"
    "-DBUILD_SHARED_LIBS:BOOL=OFF"
    "-DDART_TESTING_TIMEOUT:STRING=600"
    "-DTrilinos_WARNINGS_AS_ERRORS_FLAGS:STRING=''"
    "-DTrilinos_ENABLE_CXX11=ON"
    #
    "-DTPL_ENABLE_MPI=ON"
    "-DMPI_EXEC_POST_NUMPROCS_FLAGS:STRING='-bind-to;numa;-map-by;numa;'"
    "-DTPL_ENABLE_BinUtils=OFF"
    "-DTPL_ENABLE_SuperLU=OFF"
    "-DTPL_ENABLE_BLAS=ON"
    "-DTPL_BLAS_LIBRARIES='blas\\;gfortran\\;gomp'"
    "-DTPL_LAPACK_LIBRARIES='lapack\\;gfortran\\;gomp'"
    "-DBLAS_INCLUDE_DIRS:PATH=${BLAS_PATH}/include"
    "-DBLAS_LIBRARY_DIRS:PATH=${BLAS_PATH}/lib"
    "-DTPL_ENABLE_LAPACK=ON"
    "-DLAPACK_INCLUDE_DIRS:PATH=${LAPACK_PATH}/include"
    "-DLAPACK_LIBRARY_DIRS:PATH=${LAPACK_PATH}/lib"
    "-DTPL_ENABLE_Boost=ON"
    "-DBoost_INCLUDE_DIRS:PATH=${BOOST_PATH}/include"
    "-DBoost_LIBRARY_DIRS:PATH=${BOOST_PATH}/lib"
    "-DTPL_ENABLE_BoostLib=ON"
    "-DBoostLib_INCLUDE_DIRS:PATH=${BOOST_PATH}/include"
    "-DBoostLib_LIBRARY_DIRS:PATH=${BOOST_PATH}/lib"
    "-DTPL_ENABLE_Netcdf=ON"
    "-DNetcdf_INCLUDE_DIRS:PATH=${NETCDF_PATH}/include"
    "-DNetcdf_LIBRARY_DIRS:PATH=${NETCDF_PATH}/lib"
    "-DTPL_Netcdf_LIBRARIES:FILEPATH='${NETCDF_PATH}/lib/libnetcdf.a\\;${PNETCDF_PATH}/lib/libpnetcdf.a\\;${HDF5_PATH}/lib/libhdf5_hl.a\\;${HDF5_PATH}/lib/libhdf5.a\\;${ZLIB_PATH}/lib/libz.a\\;dl'"
    "-DTPL_ENABLE_HDF5=ON"
    "-DHDF5_INCLUDE_DIRS:PATH=${HDF5_PATH}/include"
    "-DTPL_HDF5_LIBRARIES:FILEPATH='${NETCDF_PATH}/lib/libnetcdf.a\\;${PNETCDF_PATH}/lib/libpnetcdf.a\\;${HDF5_PATH}/lib/libhdf5_hl.a\\;${HDF5_PATH}/lib/libhdf5.a\\;${ZLIB_PATH}/lib/libz.a\\;dl'"
    "-DTPL_ENABLE_Zlib=ON"
    "-DZlib_INCLUDE_DIRS:PATH=${ZLIB_PATH}/include"
    "-DTPL_Zlib_LIBRARIES:PATH=${ZLIB_PATH}/lib/libz.a"
    #
    "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF"
    "-DTrilinos_VERBOSE_CONFIGURE:BOOL=OFF"
    "-DTrilinos_ENABLE_ALL_PACKAGES:BOOL=OFF"
    "-DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF"
    "-DTrilinos_ENABLE_SECONDARY_TESTED_CODE:BOOL=ON"
    "-DTrilinos_ENABLE_EXPORT_MAKEFILES:BOOL=OFF"
    "-DTrilinos_ASSERT_MISSING_PACKAGES:BOOL=OFF"
    "-DTrilinos_WARNINGS_AS_ERRORS_FLAGS:STRING=''"
    "-DTeuchos_ENABLE_LONG_LONG_INT:BOOL=ON"
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
    "-DPhalanx_ENABLE_TESTS:BOOL=ON"
    "-DPhalanx_ENABLE_EXAMPLES:BOOL=ON"
    "-DTrilinos_ENABLE_Intrepid:BOOL=ON"
    "-DTrilinos_ENABLE_Intrepid2:BOOL=ON"
    "-DIntrepid2_ENABLE_TESTS:BOOL=ON"
    "-DIntrepid2_ENABLE_EXAMPLES:BOOL=ON"
    "-DTrilinos_ENABLE_ROL:BOOL=ON"
    "- Trilinos_ENABLE_NOX:BOOL=ON"
    "-DTrilinos_ENABLE_Stratimikos:BOOL=ON"
    "-DTrilinos_ENABLE_Thyra:BOOL=ON"
    "-DTrilinos_ENABLE_Rythmos:BOOL=ON"
    "-DTrilinos_ENABLE_OptiPack:BOOL=ON"
    "-DTrilinos_ENABLE_GlobiPack:BOOL=ON"
    "-DTrilinos_ENABLE_MOOCHO:BOOL=ON"
    "-DTrilinos_ENABLE_Stokhos:BOOL=OFF"
    "-DTrilinos_ENABLE_Piro:BOOL=ON"
    "-DTrilinos_ENABLE_Pamgen:BOOL=ON"
    "-DTrilinos_ENABLE_Isorropia:BOOL=ON"
    "-DTrilinos_ENABLE_Teko:BOOL=ON"
    "-DTrilinos_ENABLE_PyTrilinos:BOOL=OFF"
    #
    "-DTrilinos_ENABLE_STK:BOOL=ON"
    "-DTrilinos_ENABLE_STKExp:BOOL=OFF"
    "-DTrilinos_ENABLE_STKClassic:BOOL=OFF"
    "-DTrilinos_ENABLE_STKDoc_tests:BOOL=OFF"
    "-DTrilinos_ENABLE_STKIO:BOOL=ON"
    "-DTrilinos_ENABLE_STKMesh:BOOL=ON"
    "-DTrilinos_ENABLE_STKSearch:BOOL=ON"
    "-DTrilinos_ENABLE_STKSearchUtil:BOOL=OFF"
    "-DTrilinos_ENABLE_STKTopology:BOOL=ON"
    "-DTrilinos_ENABLE_STKTransfer:BOOL=ON"
    "-DTrilinos_ENABLE_STKUnit_tests:BOOL=OFF"
    "-DTrilinos_ENABLE_STKUtil:BOOL=ON"
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
    "-DStokhos_ENABLE_TEUCHOS_TIME_MONITOR:BOOL=OFF"
    "-DStratimikos_ENABLE_TEUCHOS_TIME_MONITOR:BOOL=ON"
    "-DTrilinos_ENABLE_MueLu:BOOL=ON"
    "-DAmesos2_ENABLE_KLU2:BOOL=ON"
    "-DAnasazi_ENABLE_RBGen:BOOL=ON"
    #
    "-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON"
    "-DTpetra_INST_INT_LONG_LONG:BOOL=ON"
    "-DTpetra_INST_INT_INT:BOOL=ON"
    "-DTpetra_INST_DOUBLE:BOOL=ON"
    "-DTpetra_INST_FLOAT:BOOL=OFF"
    "-DTpetra_INST_COMPLEX_FLOAT:BOOL=OFF"
    "-DTpetra_INST_COMPLEX_DOUBLE:BOOL=OFF"
    "-DTpetra_INST_INT_LONG:BOOL=OFF"
    "-DTpetra_INST_INT_UNSIGNED:BOOL=OFF"
    #
    "-DTrilinos_ENABLE_Kokkos:BOOL=ON"
    "-DTrilinos_ENABLE_KokkosCore:BOOL=ON"
    "-DPhalanx_KOKKOS_DEVICE_TYPE:STRING='OPENMP'"
    "-DPhalanx_INDEX_SIZE_TYPE:STRING='INT'"
    "-DPhalanx_SHOW_DEPRECATED_WARNINGS:BOOL=OFF"
    "-DTrilinos_ENABLE_OpenMP:BOOL=ON"
    "-DHAVE_INTREPID_KOKKOSCORE:BOOL=ON"
    "-DTPL_ENABLE_HWLOC:STRING=OFF"
    "-DTrilinos_ENABLE_ThreadPool:BOOL=OFF"
    #
    "-DTrilinos_ENABLE_Panzer:BOOL=OFF"
    "-DPanzer_ENABLE_TESTS:BOOL=ON"
    "-DPanzer_ENABLE_EXAMPLES:BOOL=ON"
    "-DPanzer_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON"
    "-DPanzer_ENABLE_FADTYPE:STRING='Sacado::Fad::DFad'"
    "-DMPI_EXEC=mpirun"
    "-DMPI_EXEC_MAX_NUMPROCS:STRING='4'"
    "-DMPI_EXEC_NUMPROCS_FLAG:STRING='-np'"
  )

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

  set_property (GLOBAL PROPERTY SubProject IKTMayerARMTrilinos)
  set_property (GLOBAL PROPERTY Label IKTMayerARMTrilinos)
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

endif()

if (BUILD_ALBANY)

  # Configure the Albany build 
  #

  set_property (GLOBAL PROPERTY SubProject IKTMayerARMAlbany)
  set_property (GLOBAL PROPERTY Label IKTMayerARMAlbany)
  
  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:FILEPATH=$ENV{jenkins_trilinos_install_dir}"
    "-DENABLE_LCM:BOOL=ON"
    "-DENABLE_MOR:BOOL=ON"
    "-DENABLE_GOAL:BOOL=OFF"
    "-DENABLE_LANDICE:BOOL=ON"
    "-DENABLE_HYDRIDE:BOOL=ON"
    "-DENABLE_AMP:BOOL=OFF"
    "-DENABLE_ATO:BOOL=ON"
    "-DENABLE_SCOREC:BOOL=OFF"
    "-DENABLE_QCAD:BOOL=ON"
    "-DENABLE_SG:BOOL=OFF"
    "-DENABLE_ENSEMBLE:BOOL=OFF"
    "-DENABLE_ASCR:BOOL=OFF"
    "-DENABLE_AERAS:BOOL=ON"
    "-DENABLE_64BIT_INT:BOOL=OFF"
    "-DENABLE_LAME:BOOL=OFF"
    "-DENABLE_DEMO_PDES:BOOL=ON"
    "-DENABLE_KOKKOS_UNDER_DEVELOPMENT:BOOL=ON"
    "-DALBANY_CTEST_TIMEOUT=500"
    "-DENABLE_CHECK_FPE:BOOL=OFF"
    "-DALBANY_MPI_EXEC_TRAILING_OPTIONS='--map-by core'"
    )
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/AlbBuild")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/AlbBuild)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuild"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit Albany configure results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message ("Cannot configure Albany build!")
  endif ()

  #
  # Build the rest of Albany and install everything
  #

  set_property (GLOBAL PROPERTY SubProject IKTMayerARMAlbany)
  set_property (GLOBAL PROPERTY Label IKTMayerARMAlbany)
  set (CTEST_BUILD_TARGET all)
  #set (CTEST_BUILD_TARGET install)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuild"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Build
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit Albany build results!")
    endif ()

  endif ()

  if (HAD_ERROR)
    message ("Cannot build Albany!")
  endif ()

  if (BUILD_LIBS_NUM_ERRORS GREATER 0)
    message ("Encountered build errors in Albany build. Exiting!")
  endif ()

  #
  # Run Albany tests
  #

  set (CTEST_TEST_TIMEOUT 500)
  CTEST_TEST (
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuild"
    RETURN_VALUE HAD_ERROR)

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Test RETURN_VALUE S_HAD_ERROR)

    if (S_HAD_ERROR)
      message ("Cannot submit Albany test results!")
    endif ()
  endif ()


endif ()
