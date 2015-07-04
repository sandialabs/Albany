cmake_minimum_required (VERSION 2.8)

if (1)
  set (CTEST_DO_SUBMIT ON)
  set (CTEST_TEST_TYPE Nightly)

  # What to build and test
  set (DOWNLOAD TRUE)
  # See if we can get away with this for speed, at least until we get onto a
  # machine that can support a lengthy nightly.
  set (CLEAN_BUILD FALSE)
  set (BUILD_SCOREC TRUE)
  set (BUILD_TRILINOS TRUE)
  set (BUILD_PERIDIGM TRUE)
  set (BUILD_ALB32 TRUE)
  set (BUILD_ALB64 FALSE)
  set (BUILD_TRILINOSCLANG11 TRUE)
  set (BUILD_ALB64CLANG11 TRUE)
  set (BUILD_ALBFUNCTOR TRUE)
  set (BUILD_INTEL_TRILINOS TRUE)
  set (BUILD_INTEL_ALBANY TRUE)
else ()
  set (CTEST_DO_SUBMIT ON)
  set (CTEST_TEST_TYPE Experimental)

  # What to build and test
  set (BUILD_ALB64 FALSE)
  set (BUILD_ALB64CLANG11 FALSE)
  set (DOWNLOAD FALSE)
  set (BUILD_SCOREC TRUE)
  set (BUILD_TRILINOS FALSE)
  set (BUILD_PERIDIGM FALSE)
  set (BUILD_ALB32 TRUE)
  set (BUILD_TRILINOSCLANG11 FALSE)
  set (CLEAN_BUILD FALSE)
  set (BUILD_ALBFUNCTOR FALSE)
  set (BUILD_INTEL_TRILINOS FALSE)
  set (BUILD_INTEL_ALBANY FALSE)
endif ()

# Begin User inputs:
set (CTEST_SITE "cee-compute011.sandia.gov" ) # generally the output of hostname
set (CTEST_DASHBOARD_ROOT "$ENV{TEST_DIRECTORY}" ) # writable path
set (CTEST_SCRIPT_DIRECTORY "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set (CTEST_CMAKE_GENERATOR "Unix Makefiles" ) # What is your compilation apps ?
set (CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?

set (INITIAL_LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})
set (PATH $ENV{PATH})

set (CTEST_PROJECT_NAME "Albany" )
set (CTEST_SOURCE_NAME repos)
set (CTEST_BUILD_NAME "linux-gcc-${CTEST_BUILD_CONFIGURATION}")
set (CTEST_BINARY_NAME build)

set (PREFIX_DIR /projects/albany)
set (GCC_MPI_DIR /sierra/sntools/SDK/mpi/openmpi/1.6.4-gcc-4.8.2-RHEL6)
set (INTEL_DIR /sierra/sntools/SDK/compilers/intel/composer_xe_2015.1.133)

set (INTEL_MPI_DIR /sierra/sntools/SDK/mpi/openmpi/1.6.4-intel-15.0-2015.2.164-RHEL6)
set (MKL_PATH /sierra/sntools/SDK/compilers/intel)

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
set (CTEST_BUILD_FLAGS "-j16")

set (CTEST_DROP_METHOD "http")

if (CTEST_DROP_METHOD STREQUAL "http")
  set (CTEST_DROP_SITE "cdash.sandia.gov")
  set (CTEST_PROJECT_NAME "Albany")
  set (CTEST_DROP_LOCATION "/CDash-2-3-0/submit.php?project=Albany")
  set (CTEST_TRIGGER_SITE "")
  set (CTEST_DROP_SITE_CDASH TRUE)
endif ()

find_program (CTEST_GIT_COMMAND NAMES git)
find_program (CTEST_SVN_COMMAND NAMES svn)

set (Trilinos_REPOSITORY_LOCATION ambradl@software.sandia.gov:/git/Trilinos)
set (SCOREC_REPOSITORY_LOCATION git@github.com:SCOREC/core.git)
set (Albany_REPOSITORY_LOCATION git@github.com:gahansen/Albany.git)
set (Peridigm_REPOSITORY_LOCATION ssh://software.sandia.gov/git/peridigm)

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

if (DOWNLOAD)
  #
  # Get the internal Trilinos repo
  #

  set (CTEST_CHECKOUT_COMMAND)

  if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/Trilinos")
    execute_process (COMMAND "${CTEST_GIT_COMMAND}" 
      clone ${Trilinos_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/Trilinos
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

  set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")

  #
  # Get the SCOREC repo
  #

  if (BUILD_SCOREC AND (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/Trilinos/SCOREC"))
    #  execute_process (COMMAND "${CTEST_SVN_COMMAND}" 
    #    checkout ${SCOREC_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/Trilinos/SCOREC
    #    OUTPUT_VARIABLE _out
    #    ERROR_VARIABLE _err
    #    RESULT_VARIABLE HAD_ERROR)
    execute_process (COMMAND "${CTEST_GIT_COMMAND}" 
      clone ${SCOREC_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/Trilinos/SCOREC
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
    
    message(STATUS "out: ${_out}")
    message(STATUS "err: ${_err}")
    message(STATUS "res: ${HAD_ERROR}")
    if (HAD_ERROR)
      message ("Cannot checkout SCOREC repository!")
      set (BUILD_SCOREC FALSE)
    endif ()
  endif ()

  #
  # Get Albany
  #

  if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/Albany")
    execute_process (COMMAND "${CTEST_GIT_COMMAND}" 
      clone ${Albany_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/Albany
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

  # Get Peridigm. Nonfatal if error.
  if (BUILD_PERIDIGM AND (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/Peridigm"))
    execute_process (COMMAND ${CTEST_GIT_COMMAND}
      clone ${Peridigm_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/Peridigm
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
    message(STATUS "out: ${_out}")
    message(STATUS "err: ${_err}")
    message(STATUS "res: ${HAD_ERROR}")
    if (HAD_ERROR)
      message (FATAL_ERROR "Cannot clone Peridigm repository.")
      set (BUILD_PERIDIGM FALSE)
    endif ()    
  endif ()
endif ()

ctest_start(${CTEST_TEST_TYPE})

#
# Send the project structure to CDash
#

if (CTEST_DO_SUBMIT)
  ctest_submit (FILES "${CTEST_SCRIPT_DIRECTORY}/Project.xml"
    RETURN_VALUE HAD_ERROR)

  if (HAD_ERROR)
    message ("Cannot submit Albany Project.xml!")
  endif ()
endif ()

if (DOWNLOAD)

  #
  # Update Trilinos
  #

  set_property (GLOBAL PROPERTY SubProject Trilinos)
  set_property (GLOBAL PROPERTY Label Trilinos)

  ctest_update(SOURCE "${CTEST_SOURCE_DIRECTORY}/Trilinos" RETURN_VALUE count)
  message("Found ${count} changed files")

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Update
      RETURN_VALUE  HAD_ERROR
      )

    if (HAD_ERROR)
      message(FATAL_ERROR "Cannot update Trilinos!")
    endif ()
  endif ()

  if (count LESS 0)
    message(FATAL_ERROR "Cannot update Trilinos!")
  endif ()


  #
  # Update the SCOREC repo
  #
  if (BUILD_SCOREC)
    set_property (GLOBAL PROPERTY SubProject SCOREC)
    set_property (GLOBAL PROPERTY Label SCOREC)

    #set (CTEST_UPDATE_COMMAND "${CTEST_SVN_COMMAND}")
    set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
    ctest_update(SOURCE "${CTEST_SOURCE_DIRECTORY}/Trilinos/SCOREC" RETURN_VALUE count)
    message("Found ${count} changed files")

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Update RETURN_VALUE  HAD_ERROR)

      if (HAD_ERROR)
        message ("Cannot update SCOREC!")
        set (BUILD_SCOREC FALSE)
      endif ()
    endif ()

    if (count LESS 0)
      message ("Cannot update SCOREC!")
      set (BUILD_SCOREC FALSE)
    endif ()
  endif ()

  #
  # Update Albany 
  #

  set_property (GLOBAL PROPERTY SubProject Albany32Bit)
  set_property (GLOBAL PROPERTY Label Albany32Bit)

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

  # Peridigm
  if (BUILD_PERIDIGM)
    set_property (GLOBAL PROPERTY SubProject Peridigm)
    set_property (GLOBAL PROPERTY Label Peridigm)

    set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
    ctest_update (SOURCE "${CTEST_SOURCE_DIRECTORY}/Peridigm" RETURN_VALUE count)
    message ("Found ${count} changed files")
    if (count LESS 0)
      set (BUILD_PERIDIGM FALSE)
    endif ()

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Update RETURN_VALUE HAD_ERROR)
    endif ()

    message ("After downloading, BUILD_PERIDIGM = ${BUILD_PERIDIGM}")
  endif ()

endif ()

#
# Set the common Trilinos config options
#

set (COMMON_CONFIGURE_OPTIONS
  "-Wno-dev"
  "-DCMAKE_BUILD_TYPE:STRING=RELEASE"
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
  "-DTPL_ENABLE_Netcdf:BOOL=ON"
  "-DNetcdf_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DNetcdf_LIBRARY_DIRS:PATH=${PREFIX_DIR}/lib"
  #
  "-DTPL_ENABLE_HDF5:BOOL=ON"
  "-DHDF5_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DHDF5_LIBRARY_DIRS:PATH=${PREFIX_DIR}/lib"
  "-DHDF5_LIBRARY_NAMES:STRING='hdf5_hl\\;hdf5\\;z'"
  #
  "-DTPL_ENABLE_Zlib:BOOL=ON"
  "-DZlib_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DZlib_LIBRARY_DIRS:PATH=${PREFIX_DIR}/lib"
  #
  "-DTPL_ENABLE_ParMETIS:BOOL=ON"
  "-DParMETIS_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DParMETIS_LIBRARY_DIRS:PATH=${PREFIX_DIR}/lib"
  #
  "-DTPL_ENABLE_SuperLU:BOOL=ON"
  "-DSuperLU_INCLUDE_DIRS:PATH=${PREFIX_DIR}/SuperLU_4.3/include"
  "-DSuperLU_LIBRARY_DIRS:PATH=${PREFIX_DIR}/SuperLU_4.3/lib"
  #
  "-DTPL_BLAS_LIBRARIES:STRING='-L${INTEL_DIR}/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_blas95_lp64 -lmkl_core -lmkl_sequential'"
  "-DTPL_LAPACK_LIBRARIES:STRING='-L${INTEL_DIR}/mkl/lib/intel64 -lmkl_lapack95_lp64'"
  #
  "-DDART_TESTING_TIMEOUT:STRING=600"
  "-DTrilinos_ENABLE_ThreadPool:BOOL=ON"
  #
  "-DTrilinos_ENABLE_TESTS:BOOL=OFF"
  "-DTrilinos_ENABLE_TriKota:BOOL=OFF"
  "-DTrilinos_ENABLE_EXPORT_MAKEFILES:BOOL=OFF"
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
  "-DTrilinos_ENABLE_Intrepid:BOOL=ON"
  "-DTrilinos_ENABLE_NOX:BOOL=ON"
  "-DTrilinos_ENABLE_Stratimikos:BOOL=ON"
  "-DTrilinos_ENABLE_Thyra:BOOL=ON"
  "-DTrilinos_ENABLE_Rythmos:BOOL=ON"
  "-DTrilinos_ENABLE_MOOCHO:BOOL=OFF"
  "-DTrilinos_ENABLE_OptiPack:BOOL=ON"
  "-DTrilinos_ENABLE_GlobiPack:BOOL=ON"
  "-DTrilinos_ENABLE_Stokhos:BOOL=ON"
  "-DTrilinos_ENABLE_Isorropia:BOOL=ON"
  "-DTrilinos_ENABLE_Piro:BOOL=ON"
  "-DTrilinos_ENABLE_Teko:BOOL=ON"
  "-DTrilinos_ENABLE_Zoltan:BOOL=ON"
  "-DTrilinos_ENABLE_Moertel:BOOL=ON"
  #
  "-DTrilinos_ENABLE_Mesquite:BOOL=OFF"
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
  "-DTrilinos_ENABLE_SEACASIoss:BOOL=ON"
  "-DTrilinos_ENABLE_SEACASExodus:BOOL=ON"
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
  #
  "-DTrilinos_ENABLE_Kokkos:BOOL=ON"
  "-DTrilinos_ENABLE_KokkosCore:BOOL=ON"
  "-DPhalanx_KOKKOS_DEVICE_TYPE:STRING=SERIAL"
  "-DPhalanx_INDEX_SIZE_TYPE:STRING=INT"
  "-DPhalanx_SHOW_DEPRECATED_WARNINGS:BOOL=OFF"
  "-DKokkos_ENABLE_Serial:BOOL=ON"
  "-DKokkos_ENABLE_OpenMP:BOOL=OFF"
  "-DKokkos_ENABLE_Pthread:BOOL=OFF"
  "-DHAVE_INTREPID_KOKKOSCORE:BOOL=ON"
  )

if (BUILD_TRILINOS)
  message ("state: BUILD_TRILINOS")

  #
  # Configure the Trilinos/SCOREC build
  #

  set_property (GLOBAL PROPERTY SubProject Trilinos)
  set_property (GLOBAL PROPERTY Label Trilinos)

  set (CONFIGURE_OPTIONS
    "-DTPL_ENABLE_MPI:BOOL=ON"
    "-DMPI_BASE_DIR:PATH=${GCC_MPI_DIR}"
    "-DCMAKE_CXX_FLAGS:STRING='-O3 -march=native -w -DNDEBUG'"
    "-DCMAKE_C_FLAGS:STRING='-O3 -march=native -w -DNDEBUG'"
    "-DCMAKE_Fortran_FLAGS:STRING='-O3 -march=native -w -DNDEBUG'"
    "-DTrilinos_EXTRA_LINK_FLAGS='-L${PREFIX_DIR}/lib -lhdf5_hl -lhdf5 -lz -lm'"
    "-DCMAKE_INSTALL_PREFIX:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "${COMMON_CONFIGURE_OPTIONS}"
    )

  if (BUILD_SCOREC)
    set (CONFIGURE_OPTIONS
      "-DTrilinos_EXTRA_REPOSITORIES:STRING=SCOREC"
      "-DTrilinos_ENABLE_SCOREC:BOOL=ON"
      "-DSCOREC_DISABLE_STRONG_WARNINGS:BOOL=ON"      
      "${CONFIGURE_OPTIONS}")
  endif ()

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
      message ("Cannot submit Trilinos/SCOREC configure results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message ("Cannot configure Trilinos/SCOREC build!")
  endif ()

  if (BUILD_SCOREC)
    #
    # SCOREC tools build inside Trilinos
    #
    # Note that we do a trick here, and just build the SCOREC_libs target, as we
    # build SCOREC as a Trilinos packages and its not possible to do that
    # independent of Trilinos. So, while this builds most of SCOREC, other
    # Trilinos capabilities are also built here.
    #

    set_property (GLOBAL PROPERTY SubProject SCOREC)
    set_property (GLOBAL PROPERTY Label SCOREC)
    set (CTEST_BUILD_TARGET "SCOREC_libs")

    MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

    CTEST_BUILD(
      BUILD "${CTEST_BINARY_DIRECTORY}/TriBuild"
      RETURN_VALUE  HAD_ERROR
      NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
      )

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Build
        RETURN_VALUE  S_HAD_ERROR
        )

      if (S_HAD_ERROR)
        message ("Cannot submit SCOREC build results!")
        set (BUILD_SCOREC FALSE)
      endif ()
    endif ()

    if (HAD_ERROR)
      message ("Cannot build SCOREC!")
      set (BUILD_SCOREC FALSE)
    endif ()

    if (BUILD_LIBS_NUM_ERRORS GREATER 0)
      message ("Encountered build errors in SCOREC build. Exiting!")
      set (BUILD_SCOREC FALSE)
    endif ()
  endif ()

  #
  # Trilinos
  #
  # Build the rest of Trilinos and install everything
  #

  set_property (GLOBAL PROPERTY SubProject Trilinos)
  set_property (GLOBAL PROPERTY Label Trilinos)
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
      message ("Cannot submit Trilinos/SCOREC build results!")
    endif ()

  endif ()

  if (HAD_ERROR)
    message ("Cannot build Trilinos!")
  endif ()

  if (BUILD_LIBS_NUM_ERRORS GREATER 0)
    message ("Encountered build errors in Trilinos build. Exiting!")
  endif ()

endif ()

if (BUILD_PERIDIGM)
  message ("state: BUILD_PERIDIGM")

  set_property (GLOBAL PROPERTY SubProject Peridigm)
  set_property (GLOBAL PROPERTY Label Peridigm)

  set (CONFIGURE_OPTIONS
    "-DCMAKE_BUILD_TYPE:STRING=Release"
    "-DENABLE_INSTALL:BOOL=ON"
    "-DCMAKE_INSTALL_PREFIX:PATH=${CTEST_BINARY_DIRECTORY}/PeridigmInstall"
    "-DTRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "-DCMAKE_C_COMPILER:STRING=mpicc"
    "-DCMAKE_CXX_COMPILER:STRING=mpicxx"
    "-DBOOST_ROOT=/projects/albany/"
    "-DUSE_DAKOTA:BOOL=OFF"
    "-DUSE_PV:BOOL=OFF"
    "-DUSE_PALS:BOOL=OFF"
    "-DCMAKE_CXX_FLAGS:STRING='-O2 -std=c++11 -Wall -pedantic -Wno-long-long -ftrapv -Wno-deprecated'"
    "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF")

  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/PeridigmBuild")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/PeridigmBuild)
  endif ()

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
    endif ()
  endif ()

  if (HAD_ERROR)
    message ("Cannot configure Peridigm build!")
    set (BUILD_PERIDIGM FALSE)
  endif ()

  if (BUILD_PERIDIGM)
    set (CTEST_BUILD_TARGET install)
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
      endif ()
    endif ()
    if (HAD_ERROR)
      message ("Cannot build Peridigm.")
      set (BUILD_PERIDIGM FALSE)
    endif ()
    if (BUILD_NUM_ERRORS GREATER 0)
      message ("Encountered build errors in Peridigm.")
      set (BUILD_PERIDIGM FALSE)
    endif ()
  endif ()

  message ("After configuring and building, BUILD_PERIDIGM = ${BUILD_PERIDIGM}")
endif ()

if (BUILD_ALB32)
  message ("state: BUILD_ALB32")

  # Configure the Albany 32 Bit build 
  # Builds everything!

  set_property (GLOBAL PROPERTY SubProject Albany32Bit)
  set_property (GLOBAL PROPERTY Label Albany32Bit)

  set (LAME_INC_DIR "/projects/sierra/linux_rh6/install/master/lame/include")
  set (LAME_LIB_DIR "/projects/sierra/linux_rh6/install/master/lame/lib")
  set (MATH_TOOLKIT_INC_DIR
    "/projects/sierra/linux_rh6/install/master/math_toolkit/include")
  set (MATH_TOOLKIT_LIB_DIR
    "/projects/sierra/linux_rh6/install/master/math_toolkit/lib")

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "-DENABLE_LCM:BOOL=ON"
    "-DENABLE_CONTACT:BOOL=ON"
    "-DENABLE_LCM_SPECULATIVE:BOOL=OFF"
    "-DENABLE_HYDRIDE:BOOL=ON"
    "-DENABLE_ENSEMBLE:BOOL=OFF"
    "-DENABLE_SG:BOOL=OFF"
    "-DENABLE_FELIX:BOOL=ON"
    "-DENABLE_AERAS:BOOL=ON"
    "-DENABLE_QCAD:BOOL=ON"
    "-DENABLE_MOR:BOOL=ON"
    "-DENABLE_ATO:BOOL=ON"
    "-DENABLE_AMP:BOOL=OFF"
    "-DENABLE_GOAL:BOOL=ON"
    "-DENABLE_ASCR:BOOL=OFF"
    "-DENABLE_CHECK_FPE:BOOL=ON"
    "-DLAME_INCLUDE_DIR:PATH=${LAME_INC_DIR}"
    "-DLAME_LIBRARY_DIR:PATH=${LAME_LIB_DIR}"
    "-DMATH_TOOLKIT_INCLUDE_DIR:PATH=${MATH_TOOLKIT_INC_DIR}"
    "-DMATH_TOOLKIT_LIBRARY_DIR:PATH=${MATH_TOOLKIT_LIB_DIR}"
    "-DENABLE_LAME:BOOL=Off") #todo
  if (BUILD_SCOREC)
    set (CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS}
      "-DENABLE_SCOREC:BOOL=ON")
  endif ()
  if (BUILD_PERIDIGM)
    set (CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS}
      "-DENABLE_PERIDIGM:BOOL=ON"
      "-DPeridigm_DIR:PATH=${CTEST_BINARY_DIRECTORY}/PeridigmInstall/lib/Peridigm/cmake")
  endif ()
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/Albany32Bit")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/Albany32Bit)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/Albany32Bit"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    APPEND
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
  # Build Albany
  #

  set (CTEST_BUILD_TARGET all)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/Albany32Bit"
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

  CTEST_TEST(
    BUILD "${CTEST_BINARY_DIRECTORY}/Albany32Bit"
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
      message ("Cannot submit Albany test results!")
    endif ()
  endif ()

  #if (HAD_ERROR)
  #	message(FATAL_ERROR "Some Albany tests failed.")
  #endif ()

endif ()

#
# Configure the Albany build using GO = long
#

if (BUILD_ALB64)
  message ("state: BUILD_ALB64")

  set_property (GLOBAL PROPERTY SubProject Albany64Bit)
  set_property (GLOBAL PROPERTY Label Albany64Bit)

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "-DENABLE_64BIT_INT:BOOL=ON"
    "-DENABLE_ALBANY_EPETRA_EXE:BOOL=OFF"
    "-DENABLE_LCM:BOOL=ON"
    "-DENABLE_GOAL:BOOL=ON"
    "-DENABLE_LCM_SPECULATIVE:BOOL=OFF"
    "-DENABLE_HYDRIDE:BOOL=ON"
    "-DENABLE_ENSEMBLE:BOOL=OFF"
    "-DENABLE_SG:BOOL=OFF"
    "-DENABLE_QCAD:BOOL=OFF"
    "-DENABLE_MOR:BOOL=OFF"
    "-DENABLE_CHECK_FPE:BOOL=ON")
  if (BUILD_SCOREC)
    set (CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS}
      "-DENABLE_SCOREC:BOOL=ON")
  endif ()

  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/Albany64Bit")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/Albany64Bit)
  endif ()

  #
  # The 64 bit build 
  #

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/Albany64Bit"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    APPEND
    )

  # Read the CTestCustom.cmake file to turn off ignored tests

  #CTEST_READ_CUSTOM_FILES("${CTEST_BINARY_DIRECTORY}/AlbanyT64")

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit Albany 64 bit configure results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message ("Cannot configure Albany 64 bit build!")
  endif ()

  #
  # Build Albany 64 bit
  #

  set (CTEST_BUILD_TARGET all)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/Albany64Bit"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Build
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit Albany 64 bit build results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message ("Cannot build Albany 64 bit!")
  endif ()

  if (BUILD_LIBS_NUM_ERRORS GREATER 0)
    message ("Encountered build errors in Albany 64 bit build. Exiting!")
  endif ()
  #
  # Run Albany 64 bit tests
  #

  CTEST_TEST(
    BUILD "${CTEST_BINARY_DIRECTORY}/Albany64Bit"
    #              PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
    #              INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
    #NUMBER_FAILED  TEST_NUM_FAILED
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Test
      RETURN_VALUE  HAD_ERROR
      )

    if (HAD_ERROR)
      message ("Cannot submit Albany 64 bit test results!")
    endif ()
  endif ()
endif ()

# Add the path to Clang libraries needed for the Clang configure, build and sest cycle
#
# Need to add the openmpi libraries at the front of LD_LIBRARY_PATH
#

set (ENV{LD_LIBRARY_PATH} 
  /projects/albany/clang/lib:${INITIAL_LD_LIBRARY_PATH}
  )

if (BUILD_TRILINOSCLANG11)
  message ("state: BUILD_TRILINOSCLANG11")
  #
  # Configure the Trilinos/SCOREC Clang build
  #

  set_property (GLOBAL PROPERTY SubProject TrilinosClang++11)
  set_property (GLOBAL PROPERTY Label TrilinosClang++11)

  set (CONFIGURE_OPTIONS
    "${COMMON_CONFIGURE_OPTIONS}"
    "-DTPL_ENABLE_MPI:BOOL=ON"
    "-DMPI_BASE_DIR:PATH=${PREFIX_DIR}/clang"
    #
    "-DTrilinos_ENABLE_CXX11:BOOL=ON"
    "-DCMAKE_CXX_FLAGS:STRING='-Os -w -DNDEBUG'"
    "-DCMAKE_C_FLAGS:STRING='-Os -w -DNDEBUG'"
    "-DCMAKE_Fortran_FLAGS:STRING='-Os -w -DNDEBUG'"
    "-DTrilinos_EXTRA_REPOSITORIES:STRING=SCOREC"
    "-DTrilinos_ENABLE_SCOREC:BOOL=ON"
    "-DSCOREC_DISABLE_STRONG_WARNINGS:BOOL=ON"
    "-DTrilinos_EXTRA_LINK_FLAGS='-L${PREFIX_DIR}/lib -lhdf5_hl -lhdf5 -lz -lm'"
    "-DCMAKE_INSTALL_PREFIX:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstallC11"
    "-DBUILD_SHARED_LIBS:BOOL=OFF" #todo Get some TPLs going with which I can do a shared build.
    "-DTPL_ENABLE_SuperLU:BOOL=OFF"
    "-DAmesos2_ENABLE_KLU2:BOOL=ON")

  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/TriBuildC11")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/TriBuildC11)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuildC11"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Trilinos"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure
      RETURN_VALUE  S_HAD_ERROR)

    if (S_HAD_ERROR)
      message ("Cannot submit TrilinosClang++11 configure results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message ("Cannot configure TrilinosClang++11 build!")
  endif ()

  #set (CTEST_BUILD_TARGET all)
  set (CTEST_BUILD_TARGET install)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuildC11"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Build
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit TrilinoClang++11 build results!")
    endif ()

  endif ()

  if (HAD_ERROR)
    message ("Cannot build Trilinos with Clang!")
  endif ()

  if (BUILD_LIBS_NUM_ERRORS GREATER 0)
    message ("Encountered build errors in Trilinos Clang build. Exiting!")
  endif ()

endif ()

#
# Configure the Albany Clang build using GO = long
#

if (BUILD_ALB64CLANG11)
  message ("state: BUILD_ALB64CLANG11")
  set_property (GLOBAL PROPERTY SubProject Albany64BitClang++11)
  set_property (GLOBAL PROPERTY Label Albany64BitClang++11)

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstallC11"
    "-DENABLE_64BIT_INT:BOOL=ON"
    "-DENABLE_ALBANY_EPETRA_EXE:BOOL=OFF"
    "-DENABLE_LCM:BOOL=ON"
    "-DENABLE_GOAL:BOOL=ON"
    "-DENABLE_QCAD:BOOL=ON"
    "-DENABLE_LCM_SPECULATIVE:BOOL=OFF"
    "-DENABLE_HYDRIDE:BOOL=ON"
    "-DENABLE_ENSEMBLE:BOOL=OFF"
    "-DENABLE_SG:BOOL=OFF"
    "-DENABLE_QCAD:BOOL=OFF"
    "-DENABLE_MOR:BOOL=OFF"
    "-DENABLE_CHECK_FPE:BOOL=ON"
    )
  if (BUILD_SCOREC)
    set (CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS}
      "-DENABLE_SCOREC:BOOL=ON")
  endif ()

  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/Albany64BitC11")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/Albany64BitC11)
  endif ()

  #
  # The Clang 64 bit build 
  #

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/Albany64BitC11"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    APPEND
    )

  # Read the CTestCustom.cmake file to turn off ignored tests

  #CTEST_READ_CUSTOM_FILES("${CTEST_BINARY_DIRECTORY}/AlbanyT64")

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message("Cannot submit Albany 64 bit Clang configure results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message("Cannot configure Albany 64 bit Clang build!")
    set (BUILD_ALB64CLANG11 FALSE)
  endif ()

  #
  # Build Clang Albany 64 bit
  #

  set (CTEST_BUILD_TARGET all)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/Albany64BitC11"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Build
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message("Cannot submit Albany 64 bit Clang build results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message("Cannot build Albany 64 bit with Clang!")
    set (BUILD_ALB64CLANG11 FALSE)
  endif ()

  if (BUILD_LIBS_NUM_ERRORS GREATER 0)
    message("Encountered build errors in Albany 64 bit Clang build. Exiting!")
    set (BUILD_ALB64CLANG11 FALSE)
  endif ()

  #
  # Run Clang Albany 64 bit tests
  #

  if (BUILD_ALB64CLANG11)
    CTEST_TEST(
      BUILD "${CTEST_BINARY_DIRECTORY}/Albany64BitC11"
      #              PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
      #              INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
      #NUMBER_FAILED  TEST_NUM_FAILED
      )

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Test
        RETURN_VALUE  HAD_ERROR
        )

      if (HAD_ERROR)
        message("Cannot submit Albany 64 bit Clang test results!")
      endif ()
    endif ()
  endif ()
endif ()

if (BUILD_ALBFUNCTOR)
  message ("state: BUILD_ALBFUNCTOR")
  # ALBANY_KOKKOS_UNDER_DEVELOPMENT build

  set_property (GLOBAL PROPERTY SubProject AlbanyFunctorDev)
  set_property (GLOBAL PROPERTY Label AlbanyFunctorDev)

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "-DENABLE_LCM:BOOL=ON"
    "-DENABLE_MOR:BOOL=ON"
    "-DENABLE_FELIX:BOOL=ON"
    "-DENABLE_HYDRIDE:BOOL=ON"
    "-DENABLE_AMP:BOOL=OFF"
    "-DENABLE_GOAL:BOOL=ON"
    "-DENABLE_ATO:BOOL=ON"
    "-DENABLE_QCAD:BOOL=ON"
    "-DENABLE_ENSEMBLE:BOOL=OFF"
    "-DENABLE_SG:BOOL=OFF"
    "-DENABLE_ASCR:BOOL=OFF"
    "-DENABLE_AERAS:BOOL=ON"
    "-DENABLE_64BIT_INT:BOOL=OFF"
    "-DENABLE_LAME:BOOL=OFF"
    "-DENABLE_DEMO_PDES:BOOL=ON"
    "-DENABLE_KOKKOS_UNDER_DEVELOPMENT:BOOL=ON"
    "-DENABLE_CHECK_FPE:BOOL=ON")
  if (BUILD_SCOREC)
    set (CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS}
      "-DENABLE_SCOREC:BOOL=ON")
  endif ()
  
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
      RETURN_VALUE HAD_ERROR
      NUMBER_ERRORS BUILD_LIBS_NUM_ERRORS
      APPEND)

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Build
        RETURN_VALUE S_HAD_ERROR)

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
    set (CTEST_TEST_TIMEOUT 120)
    CTEST_TEST (
      BUILD "${CTEST_BINARY_DIRECTORY}/AlbanyFunctorDev"
      RETURN_VALUE HAD_ERROR)

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Test RETURN_VALUE S_HAD_ERROR)

      if (S_HAD_ERROR)
        message ("Cannot submit Albany test results!")
      endif ()
    endif ()
  endif ()
endif ()

# Intel

if (BUILD_INTEL_TRILINOS)
  message ("state: BUILD_INTEL_TRILINOS")
  set_property (GLOBAL PROPERTY SubProject TrilinosIntel)
  set_property (GLOBAL PROPERTY Label TrilinosIntel)

  set (ENV{LM_LICENSE_FILE} 7500@sitelicense.sandia.gov)
  set (ENV{PATH}
    /sierra/sntools/SDK/compilers/intel/composer_xe_2015.2.164/bin/intel64:${PATH})
  set (ENV{LD_LIBRARY_PATH}
    /sierra/sntools/SDK/compilers/intel/composer_xe_2015.2.164/compiler/lib/intel64:/sierra/sntools/SDK/mpi/openmpi/1.6.4-intel-15.0-2015.2.164-RHEL6/lib:${INITIAL_LD_LIBRARY_PATH})

  set (LABLAS_LIBRARIES "-L${MKL_PATH}/lib/intel64 -Wl,--start-group ${MKL_PATH}/mkl/lib/intel64/libmkl_intel_lp64.a ${MKL_PATH}/mkl/lib/intel64/libmkl_core.a ${MKL_PATH}/mkl/lib/intel64/libmkl_sequential.a -Wl,--end-group")
  set (CONFIGURE_OPTIONS
    "${COMMON_CONFIGURE_OPTIONS}"
    "-DTPL_ENABLE_SuperLU:STRING=ON"
    "-DTPL_ENABLE_MPI:BOOL=ON"
    "-DMPI_BASE_DIR:PATH=${INTEL_MPI_DIR}"
    "-DCMAKE_CXX_FLAGS:STRING='-O3 -march=native -DNDEBUG'"
    "-DCMAKE_C_FLAGS:STRING='-O3 -march=native -DNDEBUG'"
    "-DCMAKE_Fortran_FLAGS:STRING='-O3 -march=native -DNDEBUG'"
    "-DTrilinos_EXTRA_LINK_FLAGS='-L${PREFIX_DIR}/lib -lnetcdf -lhdf5_hl -lhdf5 -lifcore -lz -Wl,-rpath,/projects/albany/lib'"
    "-DCMAKE_INSTALL_PREFIX:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosIntelInstall"
    "-DTPL_BLAS_LIBRARIES:STRING=${LABLAS_LIBRARIES}"
    "-DTPL_LAPACK_LIBRARIES:STRING=${LABLAS_LIBRARIES}"
    )

  if (BUILD_SCOREC)
    set (CONFIGURE_OPTIONS
      "-DTrilinos_EXTRA_REPOSITORIES:STRING=SCOREC"
      "-DTrilinos_ENABLE_SCOREC:BOOL=ON"
      "-DSCOREC_DISABLE_STRONG_WARNINGS:BOOL=ON"
      "-DTrilinos_ENABLE_EXPORT_MAKEFILES:BOOL=OFF"
      "-DTrilinos_ASSERT_MISSING_PACKAGES:BOOL=OFF"
      "${CONFIGURE_OPTIONS}")
  endif ()

  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/TrilinosIntel")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/TrilinosIntel)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/TrilinosIntel"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Trilinos"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR)

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure RETURN_VALUE S_HAD_ERROR)

    if (S_HAD_ERROR)
      message ("Cannot submit Trilinos/SCOREC configure results.")
    endif ()
  endif ()

  if (HAD_ERROR)
    message ("Cannot configure TrilinosIntel build.")
    set (BUILD_INTEL_ALBANY FALSE)
  endif ()

  if (BUILD_INTEL_TRILINOS)
    set (CTEST_BUILD_TARGET install)

    message ("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

    ctest_build (
      BUILD "${CTEST_BINARY_DIRECTORY}/TrilinosIntel"
      RETURN_VALUE HAD_ERROR
      NUMBER_ERRORS BUILD_LIBS_NUM_ERRORS
      APPEND)

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Build RETURN_VALUE S_HAD_ERROR)

      if (S_HAD_ERROR)
        message ("Cannot submit TrilinosIntel build results.")
      endif ()

    endif ()

    if (HAD_ERROR)
      message ("Cannot build Trilinos.")
      set (BUILD_INTEL_ALBANY FALSE)
    endif ()

    if (BUILD_LIBS_NUM_ERRORS GREATER 0)
      message ("Encountered build errors in Trilinos build. Exiting.")
      set (BUILD_INTEL_ALBANY FALSE)
    endif ()
  endif ()
endif ()

if (BUILD_INTEL_ALBANY)
  message ("state: BUILD_INTEL_ALBANY")
  set_property (GLOBAL PROPERTY SubProject AlbanyIntel)
  set_property (GLOBAL PROPERTY Label AlbanyIntel)

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosIntelInstall"
    "-DENABLE_LCM:BOOL=ON"
    "-DENABLE_MOR:BOOL=ON"
    "-DENABLE_FELIX:BOOL=ON"
    "-DENABLE_HYDRIDE:BOOL=ON"
    "-DENABLE_BGL:BOOL=OFF"
    "-DENABLE_AMP:BOOL=OFF"
    "-DENABLE_GOAL:BOOL=ON"
    "-DENABLE_ATO:BOOL=ON"
    "-DENABLE_QCAD:BOOL=ON"
    "-DENABLE_ENSEMBLE:BOOL=OFF"
    "-DENABLE_SG:BOOL=OFF"
    "-DENABLE_ASCR:BOOL=OFF"
    "-DENABLE_AERAS:BOOL=ON"
    "-DENABLE_64BIT_INT:BOOL=OFF"
    "-DENABLE_LAME:BOOL=OFF"
    "-DENABLE_DEMO_PDES:BOOL=ON"
    "-DENABLE_CHECK_FPE:BOOL=OFF")
  if (BUILD_SCOREC)
    set (CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS}
      "-DENABLE_SCOREC:BOOL=ON")
  endif ()
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/AlbanyIntel")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/AlbanyIntel)
  endif ()

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
      set (BUILD_INTEL_ALBANY FALSE)
    endif ()
  endif ()

  if (HAD_ERROR)
    message ("Cannot configure Albany build.")
    set (BUILD_INTEL_ALBANY FALSE)
  endif ()

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
        set (BUILD_INTEL_ALBANY FALSE)
      endif ()
    endif ()

    if (HAD_ERROR)
      message ("Cannot build Albany.")
      set (BUILD_INTEL_ALBANY FALSE)
    endif ()

    if (BUILD_LIBS_NUM_ERRORS GREATER 0)
      message ("Encountered build errors in Albany build.")
      set (BUILD_INTEL_ALBANY FALSE)
    endif ()
  endif ()

  if (BUILD_INTEL_ALBANY)
    set (CTEST_TEST_TIMEOUT 120)
    CTEST_TEST (
      BUILD "${CTEST_BINARY_DIRECTORY}/AlbanyIntel"
      RETURN_VALUE HAD_ERROR)

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Test RETURN_VALUE S_HAD_ERROR)

      if (S_HAD_ERROR)
        message ("Cannot submit Albany test results.")
      endif ()
    endif ()
  endif ()
endif ()
