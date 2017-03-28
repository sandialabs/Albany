# Begin User inputs:
set (CTEST_SITE "algol.ca.sandia.gov" ) # generally the output of hostname
set (CTEST_DASHBOARD_ROOT "$ENV{TEST_DIRECTORY}" ) # writable path
set (CTEST_SCRIPT_DIRECTORY "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set (CTEST_CMAKE_GENERATOR "Unix Makefiles" ) # What is your compilation app ?
set (CTEST_BUILD_CONFIGURATION release) # What type of build do you want ?

set (INITIAL_LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})

set (CTEST_PROJECT_NAME "Albany" )
set (CTEST_SOURCE_NAME repos)
set (CTEST_BUILD_NAME "linux-serial-gcc-${CTEST_BUILD_CONFIGURATION}")
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
set (CTEST_CMAKE_COMMAND "${PREFIX_DIR}/bin/cmake")
set (CTEST_COMMAND "${PREFIX_DIR}/bin/ctest -D ${CTEST_TEST_TYPE}")
set (CTEST_BUILD_FLAGS "-j72")

set (CTEST_DROP_METHOD "http")

if (CTEST_DROP_METHOD STREQUAL "http")
  set (CTEST_DROP_SITE "cdash.sandia.gov")
  set (CTEST_PROJECT_NAME "Albany")
  set (CTEST_DROP_LOCATION "/CDash-2-3-0/submit.php?project=Albany")
  set (CTEST_TRIGGER_SITE "")
  set (CTEST_DROP_SITE_CDASH TRUE)
endif ()

find_program (CTEST_GIT_COMMAND NAMES git)

set (Albany_REPOSITORY_LOCATION git@github.com:gahansen/Albany.git)
set (Trilinos_REPOSITORY_LOCATION git@github.com:trilinos/Trilinos.git)

if (CLEAN_BUILD)
  # Initial cache info
  set (CACHE_CONTENTS "
  SITE:STRING=${CTEST_SITE}
  CMAKE_BUILD_TYPE:STRING=Release
  CMAKE_GENERATOR:INTERNAL=${CTEST_CMAKE_GENERATOR}
  BUILD_TESTING:BOOL=OFF
  PRODUCT_REPO:STRING=${Albany_REPOSITORY_LOCATION}
  " )

  ctest_empty_binary_directory( "${CTEST_BINARY_DIRECTORY}" )
  file(WRITE "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt" "${CACHE_CONTENTS}")
endif ()

ctest_start(${CTEST_TEST_TYPE})

#
# Send the project structure to CDash
#

if (CTEST_DO_SUBMIT)
  ctest_submit (FILES "${CTEST_SCRIPT_DIRECTORY}/Project.xml"
    RETURN_VALUE  HAD_ERROR
    )

  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot submit Albany Project.xml!")
  endif ()
endif ()

if (DOWNLOAD_TRILINOS)

  set (CTEST_CHECKOUT_COMMAND)

  #
  #  Get Trilinos
  #  
  if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/Trilinos")
    execute_process (COMMAND "${CTEST_GIT_COMMAND}"
      clone ${Trilinos_REPOSITORY_LOCATION} -b master ${CTEST_SOURCE_DIRECTORY}/Trilinos
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

endif()

    


if (DOWNLOAD_ALBANY)

  #
  # Update Albany 
  #

  set_property (GLOBAL PROPERTY SubProject LCMAlbanyReleaseAlgol)
  set_property (GLOBAL PROPERTY Label LCMAlbanyReleaseAlgol)

  set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
  CTEST_UPDATE(SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany" RETURN_VALUE count)
  message("Found ${count} changed files")

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Update
      RETURN_VALUE  HAD_ERROR
      )

    if (HAD_ERROR)
      message(FATAL_ERROR "Cannot update Albany repository!")
    endif ()
  endif ()

  if (count LESS 0)
    message(FATAL_ERROR "Cannot update Albany!")
  endif ()

endif ()

if (BUILD_TRILINOS_LCM_RELEASE)

  message ("ctest state: BUILD_TRILINOS_SERIAL")
  #
  #  Configure Trilinos
  #
  set_property (GLOBAL PROPERTY SubProject LCMTrilinosReleaseAlgol)
  set_property (GLOBAL PROPERTY Label LCMTrilinosReleaseAlgol)

  set (CONFIGURE_OPTIONS
    "-DBUILD_SHARED_LIBS:BOOL=ON"
    "-DCMAKE_BUILD_TYPE:STRING='RELEASE'"
    "-DCMAKE_CXX_COMPILER:FILEPATH=/usr/lib64/openmpi/bin/mpicxx"
    "-DCMAKE_C_COMPILER:FILEPATH=/usr/lib64/openmpi/bin/mpicc"
    "-DCMAKE_Fortran_COMPILER:FILEPATH=/usr/lib64/openmpi/bin/mpif90"
    "-DCMAKE_INSTALL_PREFIX:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF"
    #
    "-DTPL_ENABLE_MPI:BOOL=ON"
    "-DTPL_MPI_INCLUDE_DIRS:FILEPATH=/usr/include/openmpi-x86_64"
    "-DTPL_MPI_LIBRARY_DIRS:FILEPATH=/usr/lib64/openmpi/lib"
    "-DMPI_BIN_DIR:FILEPATH=/usr/lib64/openmpi/bin"
    #
    "-DTPL_ENABLE_Boost:BOOL=ON"
    "-DTPL_ENABLE_BoostLib:BOOL=ON"
    "-DBoostLib_INCLUDE_DIRS:FILEPATH=/usr/include/boost"
    "-DBoostLib_LIBRARY_DIRS:FILEPATH=/usr/lib64"
    "-DBoost_INCLUDE_DIRS:FILEPATH=/usr/include/boost"
    #
    "-DTrilinos_DISABLE_ENABLED_FORWARD_DEP_PACKAGES=ON"
    "-DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF"
    "-DTrilinos_ENABLE_ALL_PACKAGES:BOOL=OFF"
    "-DTrilinos_ENABLE_CXX11:BOOL=ON"
    "-DTrilinos_ENABLE_EXAMPLES:BOOL=OFF"
    "-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON"
    "-DTrilinos_VERBOSE_CONFIGURE:BOOL=OFF"
    "-DTrilinos_WARNINGS_AS_ERRORS_FLAGS:STRING=''"
    #
    "-DHAVE_INTREPID_KOKKOSCORE:BOOL=ON"
    "-DKokkos_ENABLE_CXX11:BOOL=ON"
    "-DKokkos_ENABLE_Cuda_UVM:BOOL=OFF"
    "-DKokkos_ENABLE_EXAMPLES:BOOL=OFF"
    "-DKokkos_ENABLE_OpenMP:BOOL=OFF"
    "-DKokkos_ENABLE_Pthread:BOOL=OFF"
    "-DKokkos_ENABLE_Serial:BOOL=ON"
    "-DKokkos_ENABLE_TESTS:BOOL=ON"
    "-DTPL_ENABLE_CUDA:STRING=OFF"
    "-DTPL_ENABLE_CUSPARSE:BOOL=OFF"
    #
    "-DAmesos2_ENABLE_KLU2:BOOL=ON"
    "-DEpetraExt_USING_HDF5:BOOL=OFF"
    "-DIntrepid2_ENABLE_TESTS:BOOL=OFF"
    "-DIntrepid2_ENABLE_KokkosDynRankView:BOOL=ON"
    "-DMiniTensor_ENABLE_TESTS:BOOL=ON"
    "-DROL_ENABLE_TESTS:BOOL=OFF"
    "-DPhalanx_INDEX_SIZE_TYPE:STRING='INT'"
    "-DPhalanx_KOKKOS_DEVICE_TYPE:STRING='SERIAL'"
    "-DPhalanx_SHOW_DEPRECATED_WARNINGS:BOOL=OFF"
    "-DSacado_ENABLE_COMPLEX:BOOL=ON"
    "-DTeuchos_ENABLE_COMPLEX:BOOL=ON"
    "-DTpetra_ENABLE_Kokkos_Refactor:BOOL=ON"
    "-DTpetra_INST_PTHREAD:BOOL=OFF"
    #
    "-DTPL_ENABLE_HDF5:BOOL=OFF"
    "-DTPL_ENABLE_HWLOC:STRING=OFF"
    "-DTPL_ENABLE_Matio:BOOL=OFF"
    "-DTPL_ENABLE_Netcdf:BOOL=ON"
    "-DTPL_ENABLE_X11:BOOL=OFF"
    "-DTPL_Netcdf_INCLUDE_DIRS:PATH=/usr/local/netcdf/include"
    "-DTPL_Netcdf_LIBRARY_DIRS:PATH=/usr/local/netcdf/lib"
    "-DTPL_Netcdf_LIBRARIES:PATH='/usr/local/netcdf/lib/libnetcdf.so'"
    "-DTPL_Netcdf_PARALLEL:BOOL=ON"
    #
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
    "-DTrilinos_ENABLE_OpenMP:BOOL=OFF"
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
    "-DTrilinos_ENABLE_Teuchos:BOOL=ON"
    "-DTrilinos_ENABLE_ThreadPool:BOOL=ON"
    "-DTrilinos_ENABLE_Thyra:BOOL=ON"
    "-DTrilinos_ENABLE_Tpetra:BOOL=ON"
    "-DTrilinos_ENABLE_Zoltan2:BOOL=ON"
    "-DTrilinos_ENABLE_Zoltan:BOOL=ON"
  )
   if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/TriBuild")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/TriBuild)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuild"
    SOURCE "/home/lcm/LCM/Trilinos"
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
  #
  set_property (GLOBAL PROPERTY SubProject LCMTrilinosReleaseAlgol)
  set_property (GLOBAL PROPERTY Label LCMTrilinosReleaseAlgol)
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



if (BUILD_ALB_LCM_RELEASE)

  # Configure Albany 
  #

  set_property (GLOBAL PROPERTY SubProject LCMAlbanyReleaseAlgol)
  set_property (GLOBAL PROPERTY Label LCMAlbanyReleaseAlgol)

  set (TRILINSTALLDIR "/home/lcm/LCM/trilinos-install-serial-gcc-release")

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:FILEPATH=${TRILINSTALLDIR}"
    "-DCMAKE_CXX_FLAGS:STRING='-msse3'"
    "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF"
    "-DENABLE_LCM:BOOL=ON"
    "-DENABLE_ATO:BOOL=OFF"
    "-DENABLE_QCAD:BOOL=OFF"
    "-DENABLE_MOR:BOOL=OFF"
    "-DENABLE_SG:BOOL=OFF"
    "-DENABLE_ENSEMBLE:BOOL=OFF"
    "-DENABLE_FELIX:BOOL=OFF"
    "-DENABLE_LAME:BOOL=OFF"
    "-DENABLE_LAMENT:BOOL=OFF"
    "-DENABLE_CHECK_FPE:BOOL=OFF"
    "-DENABLE_FLUSH_DENORMALS:BOOL=ON"
    "-DENABLE_KOKKOS_UNDER_DEVELOPMENT:BOOL=OFF"
    "-DALBANY_ENABLE_FORTRAN:BOOL=OFF"
    "-DENABLE_SLFAD:BOOL=OFF"
  ) 
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/LCMAlbanyReleaseAlgol")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/LCMAlbanyReleaseAlgol)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/LCMAlbanyReleaseAlgol"
    SOURCE "/home/lcm/LCM/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit Albany configure results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot configure Albany build!")
  endif ()

  #
  # Build Albany
  #

  set (CTEST_BUILD_TARGET all)
  #set (CTEST_BUILD_TARGET install)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/LCMAlbanyReleaseAlgol"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Build
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit Albany build results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot build Albany!")
  endif ()

  if (BUILD_LIBS_NUM_ERRORS GREATER 0)
    message(FATAL_ERROR "Encountered build errors in Albany build. Exiting!")
  endif ()

  #
  # Run Albany tests
  #

  CTEST_TEST(
    BUILD "${CTEST_BINARY_DIRECTORY}/LCMAlbanyReleaseAlgol"
    #              PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
    #              INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
    #NUMBER_FAILED  TEST_NUM_FAILED
    RETURN_VALUE  HAD_ERROR
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Test
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit Albany test results!")
    endif ()
  endif ()

  #if (HAD_ERROR)
  #	message(FATAL_ERROR "Some Albany tests failed.")
  #endif ()

endif ()
