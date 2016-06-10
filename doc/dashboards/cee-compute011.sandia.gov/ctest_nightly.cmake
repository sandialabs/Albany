cmake_minimum_required (VERSION 2.8)

SET(CTEST_DO_SUBMIT "$ENV{DO_SUBMIT}")
SET(CTEST_TEST_TYPE "$ENV{TEST_TYPE}")
SET(CTEST_BUILD_OPTION "$ENV{BUILD_OPTION}")

if (1)
  # What to build and test
  IF(CTEST_BUILD_OPTION MATCHES "base")
    # Only download repos and cleanout in the base nightly test run (start it an hour earlier)
    set (CLEAN_BUILD TRUE)
    set (DOWNLOAD TRUE)
  ELSE()
    set (CLEAN_BUILD FALSE)
    set (DOWNLOAD FALSE)
  ENDIF()

  set (BUILD_SCOREC TRUE)
  set (BUILD_TRILINOS TRUE)
  set (BUILD_PERIDIGM TRUE)
  set (BUILD_ALB32 TRUE)
  set (BUILD_ALB64 TRUE)
  set (BUILD_ALBFUNCTOR TRUE)
  IF(CTEST_BUILD_OPTION MATCHES "clang")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_PERIDIGM FALSE)
    set (BUILD_ALB32 FALSE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_TRILINOSCLANG TRUE)
    set (BUILD_ALB64CLANG TRUE)
    set (BUILD_ALBFUNCTOR FALSE)
    set (BUILD_INTEL_TRILINOS FALSE)
    set (BUILD_INTEL_ALBANY FALSE)
  ELSE()
    set (BUILD_TRILINOSCLANG FALSE)
    set (BUILD_ALB64CLANG FALSE)
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "intel")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_PERIDIGM FALSE)
    set (BUILD_ALB32 FALSE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_TRILINOSCLANG FALSE)
    set (BUILD_ALB64CLANG FALSE)
    set (BUILD_ALBFUNCTOR FALSE)
    set (BUILD_INTEL_TRILINOS TRUE)
    set (BUILD_INTEL_ALBANY TRUE)
  ELSE()
    set (BUILD_INTEL_TRILINOS FALSE)
    set (BUILD_INTEL_ALBANY FALSE)
  ENDIF()
else ()
  # This block is for testing. Set "if (1)" to "if (0)", and then freely mess
  # around with the settings in this block.

  # What to build and test
  set (DOWNLOAD TRUE)
  # See if we can get away with this for speed, at least until we get onto a
  # machine that can support a lengthy nightly.
  set (CLEAN_BUILD TRUE)
  set (BUILD_SCOREC TRUE)
  set (BUILD_TRILINOS TRUE)
  set (BUILD_PERIDIGM TRUE)
  set (BUILD_ALB32 TRUE)
  set (BUILD_ALB64 TRUE)
  set (BUILD_TRILINOSCLANG TRUE)
  set (BUILD_ALB64CLANG TRUE)
  set (BUILD_ALBFUNCTOR TRUE)
  set (BUILD_INTEL_TRILINOS TRUE)
  set (BUILD_INTEL_ALBANY TRUE)
endif ()

set (extra_cxx_flags "")

# Begin User inputs:
set (CTEST_SITE "cee-compute011.sandia.gov" ) # generally the output of hostname
set (CTEST_DASHBOARD_ROOT "$ENV{TEST_DIRECTORY}" ) # writable path
set (CTEST_SCRIPT_DIRECTORY "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set (CTEST_CMAKE_GENERATOR "Unix Makefiles" ) # What is your compilation apps ?
set (CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?

set (INITIAL_LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})
#set (PATH $ENV{PATH})

set (CTEST_PROJECT_NAME "Albany" )
set (CTEST_SOURCE_NAME repos)
set (CTEST_BUILD_NAME "linux-gcc-${CTEST_BUILD_CONFIGURATION}")
set (CTEST_BINARY_NAME build)

set (PREFIX_DIR /projects/albany)
set (GCC_MPI_DIR /sierra/sntools/SDK/mpi/openmpi/1.8.8-gcc-5.2.0-RHEL6)
set (INTEL_DIR /sierra/sntools/SDK/compilers/intel/composer_xe_2016.3.210)

#set (BOOST_ROOT /projects/albany/nightly)
set (BOOST_ROOT /projects/albany)

set (INTEL_MPI_DIR /sierra/sntools/SDK/mpi/openmpi/1.8.8-intel-16.0-2016.3.210-RHEL6)
#set (MKL_PATH /sierra/sntools/SDK/compilers/intel)
set (MKL_PATH /sierra/sntools/SDK/compilers/intel/composer_xe_2016.3.210)

set (USE_LAME OFF)
set (LAME_INC_DIR "/projects/sierra/linux_rh6/install/master/lame/include\;/projects/sierra/linux_rh6/install/master/Sierra/sierra_util/include\;/projects/sierra/linux_rh6/install/master/stk/stk_expreval/include\;/projects/sierra/linux_rh6/install/master/utility/include\;/projects/sierra/linux_rh6/install/master/Sierra/include")
set (LAME_LIB_DIR "/projects/sierra/linux_rh6/install/master/lame/lib\;/projects/sierra/linux_rh6/install/master/Sierra/sierra_util/lib\;/projects/sierra/linux_rh6/install/master/stk/stk_expreval/lib\;/projects/sierra/linux_rh6/install/master/utility/lib\;/projects/sierra/linux_rh6/install/master/Sierra/lib")
set (LAME_LIBRARIES "sierra_util_diag\;sierra_util_events\;sierra_util_user_input_function\;sierra_util_domain\;sierra_util_sctl\;stk_expreval\;utility\;sierra\;dataManager\;audit\;sierraparser")
set (MATH_TOOLKIT_INC_DIR
  "/projects/sierra/linux_rh6/install/master/math_toolkit/include")
set (MATH_TOOLKIT_LIB_DIR
  "/projects/sierra/linux_rh6/install/master/math_toolkit/lib")

set (CTEST_SOURCE_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_SOURCE_NAME}")
set (CTEST_BINARY_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_BINARY_NAME}")

IF (CLEAN_BUILD)
  IF(EXISTS "${CTEST_BINARY_DIRECTORY}" )
    FILE(REMOVE_RECURSE "${CTEST_BINARY_DIRECTORY}")
  ENDIF()
ENDIF()

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

#set (Trilinos_REPOSITORY_LOCATION https://github.com/trilinos/Trilinos.git)
set (Trilinos_REPOSITORY_LOCATION git@github.com:trilinos/Trilinos.git)
set (SCOREC_REPOSITORY_LOCATION git@github.com:SCOREC/core.git)
set (Albany_REPOSITORY_LOCATION git@github.com:gahansen/Albany.git)
#set (Peridigm_REPOSITORY_LOCATION https://github.com/peridigm/peridigm) #ssh://software.sandia.gov/git/peridigm)
set (Peridigm_REPOSITORY_LOCATION git@github.com:peridigm/peridigm) #ssh://software.sandia.gov/git/peridigm)

if (CLEAN_BUILD)
  # Initial cache info
  set (CACHE_CONTENTS "
  SITE:STRING=${CTEST_SITE}
  CMAKE_BUILD_TYPE:STRING=Release
  CMAKE_GENERATOR:INTERNAL=${CTEST_CMAKE_GENERATOR}
  BUILD_TESTING:BOOL=OFF
  PRODUCT_REPO:STRING=${Albany_REPOSITORY_LOCATION}
  " )

#  ctest_empty_binary_directory( "${CTEST_BINARY_DIRECTORY}" )
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
  "-DMDS_ID_TYPE:STRING='long int'"
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
  "-DBoost_INCLUDE_DIRS:PATH=${BOOST_ROOT}/include"
  "-DBoost_LIBRARY_DIRS:PATH=${BOOST_ROOT}/lib"
  "-DBoostLib_INCLUDE_DIRS:PATH=${BOOST_ROOT}/include"
  "-DBoostLib_LIBRARY_DIRS:PATH=${BOOST_ROOT}/lib"
  "-DBoostAlbLib_INCLUDE_DIRS:PATH=${BOOST_ROOT}/include"
  "-DBoostAlbLib_LIBRARY_DIRS:PATH=${BOOST_ROOT}/lib"
  #
  "-DTPL_ENABLE_Netcdf:BOOL=ON"
  "-DNetcdf_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DNetcdf_LIBRARY_DIRS:PATH=${PREFIX_DIR}/lib"
  "-DTPL_Netcdf_PARALLEL:BOOL=ON"
  "-DTPL_ENABLE_Pnetcdf:STRING=ON"
  "-DPnetcdf_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DPnetcdf_LIBRARY_DIRS=${PREFIX_DIR}/lib"
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
  "-DTrilinos_ENABLE_Intrepid2:BOOL=ON"
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
  "-DTrilinos_ENABLE_Moertel:BOOL=ON"
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
  )

INCLUDE(${CTEST_SCRIPT_DIRECTORY}/trilinos_macro.cmake)

if (BUILD_TRILINOS)

  set (CONF_OPTS
    "-DTPL_ENABLE_MPI:BOOL=ON"
    "-DMPI_BASE_DIR:PATH=${GCC_MPI_DIR}"
    "-DCMAKE_CXX_COMPILER:STRING=${GCC_MPI_DIR}/bin/mpicxx"
    "-DCMAKE_CXX_FLAGS:STRING='-O3 -march=native -w -DNDEBUG ${extra_cxx_flags}'"
    "-DCMAKE_C_COMPILER:STRING=${GCC_MPI_DIR}/bin/mpicc"
    "-DCMAKE_C_FLAGS:STRING='-O3 -march=native -w -DNDEBUG'"
    "-DCMAKE_Fortran_COMPILER:STRING=${GCC_MPI_DIR}/bin/mpifort"
    "-DCMAKE_Fortran_FLAGS:STRING='-O3 -march=native -w -DNDEBUG'"
    "-DTrilinos_EXTRA_LINK_FLAGS='-L${PREFIX_DIR}/lib -lnetcdf -lpnetcdf -lhdf5_hl -lhdf5 -lz -lm'"
    "-DCMAKE_INSTALL_PREFIX:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "${COMMON_CONFIGURE_OPTIONS}"
    )

    if (BUILD_SCOREC)
      set (CONF_OPTS
        "-DTrilinos_ENABLE_SCOREC:BOOL=ON"
        "-DSCOREC_DISABLE_STRONG_WARNINGS:BOOL=ON"      
        "${CONF_OPTS}")
    endif (BUILD_SCOREC)

  do_trilinos("${CONF_OPTS}" "TrilinosBld")

endif (BUILD_TRILINOS)

if (BUILD_PERIDIGM)
  INCLUDE(${CTEST_SCRIPT_DIRECTORY}/peridigm_macro.cmake)
  do_peridigm()
endif (BUILD_PERIDIGM)

INCLUDE(${CTEST_SCRIPT_DIRECTORY}/albany_macro.cmake)

if (BUILD_ALB32)

  set (CONF_OPTIONS
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
    "-DENABLE_ASCR:BOOL=OFF"
    "-DENABLE_CHECK_FPE:BOOL=ON")
  if (BUILD_SCOREC)
    set (CONF_OPTIONS ${CONF_OPTIONS}
      "-DENABLE_SCOREC:BOOL=ON"
      "-DENABLE_GOAL:BOOL=ON")
  endif (BUILD_SCOREC)
  if (BUILD_PERIDIGM)
    set (CONF_OPTIONS ${CONF_OPTIONS}
      "-DENABLE_PERIDIGM:BOOL=ON"
      "-DPeridigm_DIR:PATH=${CTEST_BINARY_DIRECTORY}/PeridigmInstall/lib/Peridigm/cmake")
  endif (BUILD_PERIDIGM)

  do_albany("${CONF_OPTIONS}" "Albany32Bit")

endif (BUILD_ALB32)

#
# Configure the Albany build using GO = long
#

if (BUILD_ALB64)

  set (CONF_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "-DENABLE_64BIT_INT:BOOL=ON"
    "-DENABLE_ALBANY_EPETRA_EXE:BOOL=OFF"
    "-DENABLE_LCM:BOOL=ON"
    "-DENABLE_LCM_SPECULATIVE:BOOL=OFF"
    "-DENABLE_HYDRIDE:BOOL=ON"
    "-DENABLE_ENSEMBLE:BOOL=OFF"
    "-DENABLE_SG:BOOL=OFF"
    "-DENABLE_QCAD:BOOL=OFF"
    "-DENABLE_MOR:BOOL=OFF"
    "-DENABLE_CHECK_FPE:BOOL=ON")
  if (BUILD_SCOREC)
    set (CONF_OPTIONS ${CONF_OPTIONS}
      "-DENABLE_SCOREC:BOOL=ON"
      "-DENABLE_GOAL:BOOL=ON")
  endif (BUILD_SCOREC)

  do_albany("${CONF_OPTIONS}" "Albany64Bit")

endif (BUILD_ALB64)

# Add the path to Clang libraries needed for the Clang configure, build and sest cycle
#
# Need to add the openmpi libraries at the front of LD_LIBRARY_PATH
#

set (ENV{LD_LIBRARY_PATH} 
  ${PREFIX_DIR}/clang/lib:${INITIAL_LD_LIBRARY_PATH}
  )

if (BUILD_TRILINOSCLANG)

  set (CONFIGURE_OPTIONS
    "${COMMON_CONFIGURE_OPTIONS}"
    "-DTPL_ENABLE_MPI:BOOL=ON"
    "-DMPI_BASE_DIR:PATH=${PREFIX_DIR}/clang"
    #
    "-DCMAKE_CXX_COMPILER:STRING=/projects/albany/clang-3.7/bin/mpicxx"
    "-DCMAKE_CXX_FLAGS:STRING='-Os -w -DNDEBUG ${extra_cxx_flags}'"
    "-DCMAKE_C_COMPILER:STRING=/projects/albany/clang-3.7/bin/mpicc"
    "-DCMAKE_C_FLAGS:STRING='-Os -w -DNDEBUG'"
    "-DCMAKE_Fortran_COMPILER:STRING=/projects/albany/clang-3.7/bin/mpifort"
    "-DCMAKE_Fortran_FLAGS:STRING='-Os -w -DNDEBUG'"
    "-DTrilinos_ENABLE_SCOREC:BOOL=ON"
    "-DMDS_ID_TYPE:STRING='long long int'"
    "-DSCOREC_DISABLE_STRONG_WARNINGS:BOOL=ON"
    "-DTrilinos_EXTRA_LINK_FLAGS='-L${PREFIX_DIR}/lib -lnetcdf -lpnetcdf -lhdf5_hl -lhdf5 -lz -lm'"
    "-DCMAKE_INSTALL_PREFIX:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstallC11"
    "-DBUILD_SHARED_LIBS:BOOL=OFF"
    "-DTPL_ENABLE_SuperLU:BOOL=OFF"
    "-DAmesos2_ENABLE_KLU2:BOOL=ON")

  do_trilinos("${CONFIGURE_OPTIONS}" "TrilinosClangBld")

endif (BUILD_TRILINOSCLANG)

#
# Configure the Albany Clang build using GO = long
#

if (BUILD_ALB64CLANG)

  set (CONF_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstallC11"
    "-DENABLE_64BIT_INT:BOOL=ON"
    "-DENABLE_ALBANY_EPETRA_EXE:BOOL=OFF"
    "-DENABLE_LCM:BOOL=ON"
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
    set (CONF_OPTIONS ${CONF_OPTIONS}
      "-DENABLE_SCOREC:BOOL=ON"
      "-DENABLE_GOAL:BOOL=ON")
  endif (BUILD_SCOREC)

  do_albany("${CONF_OPTIONS}" "Albany64BitClang")

endif (BUILD_ALB64CLANG)

if (BUILD_ALBFUNCTOR)

  set (CONF_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "-DENABLE_LCM:BOOL=ON"
    "-DENABLE_MOR:BOOL=ON"
    "-DENABLE_FELIX:BOOL=ON"
    "-DENABLE_HYDRIDE:BOOL=ON"
    "-DENABLE_AMP:BOOL=OFF"
    "-DENABLE_ATO:BOOL=ON"
    "-DENABLE_QCAD:BOOL=ON"
    "-DENABLE_ENSEMBLE:BOOL=OFF"
    "-DENABLE_SG:BOOL=OFF"
    "-DENABLE_ASCR:BOOL=OFF"
    "-DENABLE_AERAS:BOOL=ON"
    "-DENABLE_64BIT_INT:BOOL=OFF"
    "-DENABLE_DEMO_PDES:BOOL=ON"
    "-DENABLE_KOKKOS_UNDER_DEVELOPMENT:BOOL=ON"
    "-DENABLE_CHECK_FPE:BOOL=ON")
  if (BUILD_SCOREC)
    set (CONF_OPTIONS ${CONF_OPTIONS}
      "-DENABLE_SCOREC:BOOL=ON"
      "-DENABLE_GOAL:BOOL=ON")
  endif (BUILD_SCOREC)

  do_albany("${CONF_OPTIONS}" "AlbanyFunctorDev")

endif (BUILD_ALBFUNCTOR)

if (BUILD_INTEL_TRILINOS)
  INCLUDE(${CTEST_SCRIPT_DIRECTORY}/intel_macro.cmake)
   do_intel()
endif (BUILD_INTEL_TRILINOS)
