cmake_minimum_required (VERSION 2.8)

SET(CTEST_DO_SUBMIT "$ENV{DO_SUBMIT}")
SET(CTEST_TEST_TYPE "$ENV{TEST_TYPE}")
SET(CTEST_BUILD_OPTION "$ENV{BUILD_OPTION}")

if (1)
  # What to build and test
  IF(CTEST_BUILD_OPTION MATCHES "base-trilinos") 
    set (DOWNLOAD_TRILINOS TRUE)
    set (DOWNLOAD_ALBANY FALSE)
    set (BUILD_TRILINOS TRUE)
    set (BUILD_ALB64 FALSE) 
  ENDIF() 
  IF(CTEST_BUILD_OPTION MATCHES "base-albany") 
    set (DOWNLOAD_TRILINOS FALSE)
    set (DOWNLOAD_ALBANY TRUE)
    set (BUILD_TRILINOS FALSE)
    set (BUILD_ALB64 TRUE) 
  ENDIF() 
  set (CLEAN_BUILD TRUE)
# Drop the gcc 32bit build
  set (BUILD_ALB32 FALSE)
# Drop the functor dev build - is this still of interest?
  set (BUILD_ALBFUNCTOR FALSE)
  set (CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?
  IF(CTEST_BUILD_OPTION MATCHES "debug-trilinos")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_TRILINOSDBG TRUE)
    set (BUILD_PERIDIGM FALSE)
    set (BUILD_ALB32 FALSE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_ALB64DBG FALSE)
    set (BUILD_TRILINOSCLANG FALSE)
    set (BUILD_ALB64CLANG FALSE)
    set (BUILD_TRILINOSCLANGDBG FALSE)
    set (BUILD_ALB64CLANGDBG FALSE)
    set (BUILD_ALBFUNCTOR FALSE)
    set (BUILD_INTEL_TRILINOS FALSE)
    set (BUILD_INTEL_ALBANY FALSE)
    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
  ELSE()
    set (BUILD_TRILINOSDBG FALSE)
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "debug-albany")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_TRILINOSDBG FALSE)
    set (BUILD_PERIDIGM FALSE)
    set (BUILD_ALB32 FALSE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_ALB64DBG TRUE)
    set (BUILD_TRILINOSCLANG FALSE)
    set (BUILD_ALB64CLANG FALSE)
    set (BUILD_TRILINOSCLANGDBG FALSE)
    set (BUILD_ALB64CLANGDBG FALSE)
    set (BUILD_ALBFUNCTOR FALSE)
    set (BUILD_INTEL_TRILINOS FALSE)
    set (BUILD_INTEL_ALBANY FALSE)
    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
  ELSE()
    set (BUILD_ALB64DBG FALSE)
  ENDIF() 
  IF(CTEST_BUILD_OPTION MATCHES "clang-trilinos")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_PERIDIGM FALSE)
    set (BUILD_ALB32 FALSE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_TRILINOSCLANG TRUE)
    set (BUILD_ALB64CLANG FALSE)
    set (BUILD_TRILINOSCLANGDBG FALSE)
    set (BUILD_ALB64CLANGDBG FALSE)
    set (BUILD_ALBFUNCTOR FALSE)
    set (BUILD_INTEL_TRILINOS FALSE)
    set (BUILD_INTEL_ALBANY FALSE)
    set (CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?
#    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
  ELSE()
    set (BUILD_TRILINOSCLANG FALSE)
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "clang-albany")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_PERIDIGM FALSE)
    set (BUILD_ALB32 FALSE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_TRILINOSCLANG FALSE)
    set (BUILD_ALB64CLANG TRUE)
    set (BUILD_TRILINOSCLANGDBG FALSE)
    set (BUILD_ALB64CLANGDBG FALSE)
    set (BUILD_ALBFUNCTOR FALSE)
    set (BUILD_INTEL_TRILINOS FALSE)
    set (BUILD_INTEL_ALBANY FALSE)
    set (CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?
#    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
  ELSE()
    set (BUILD_ALB64CLANG FALSE)
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "clangdbg-trilinos")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_PERIDIGM FALSE)
    set (BUILD_ALB32 FALSE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_TRILINOSCLANG FALSE)
    set (BUILD_ALB64CLANG FALSE)
    set (BUILD_TRILINOSCLANGDBG TRUE)
    set (BUILD_ALB64CLANGDBG FALSE)
    set (BUILD_ALBFUNCTOR FALSE)
    set (BUILD_INTEL_TRILINOS FALSE)
    set (BUILD_INTEL_ALBANY FALSE)
    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
#    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
  ELSE()
    set (BUILD_TRILINOSCLANGDBG FALSE)
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "clangdbg-albany")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_PERIDIGM FALSE)
    set (BUILD_ALB32 FALSE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_TRILINOSCLANG FALSE)
    set (BUILD_ALB64CLANG FALSE)
    set (BUILD_TRILINOSCLANGDBG FALSE)
    set (BUILD_ALB64CLANGDBG TRUE)
    set (BUILD_ALBFUNCTOR FALSE)
    set (BUILD_INTEL_TRILINOS FALSE)
    set (BUILD_INTEL_ALBANY FALSE)
    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
#    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
  ELSE()
    set (BUILD_ALB64CLANGDBG FALSE)
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "intel-trilinos")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_PERIDIGM FALSE)
    set (BUILD_ALB32 FALSE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_TRILINOSCLANG FALSE)
    set (BUILD_ALB64CLANG FALSE)
    set (BUILD_TRILINOSCLANGDBG FALSE)
    set (BUILD_ALB64CLANGDBG FALSE)
    set (BUILD_ALBFUNCTOR FALSE)
    set (BUILD_INTEL_TRILINOS TRUE)
    set (BUILD_INTEL_ALBANY FALSE)
    set (CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?
#    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
  ELSE()
    set (BUILD_INTEL_TRILINOS FALSE)
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "intel-albany")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_PERIDIGM FALSE)
    set (BUILD_ALB32 FALSE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_TRILINOSCLANG FALSE)
    set (BUILD_ALB64CLANG FALSE)
    set (BUILD_TRILINOSCLANGDBG FALSE)
    set (BUILD_ALB64CLANGDBG FALSE)
    set (BUILD_ALBFUNCTOR FALSE)
    set (BUILD_INTEL_TRILINOS FALSE)
    set (BUILD_INTEL_ALBANY TRUE)
    set (CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?
#    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
  ELSE()
    set (BUILD_INTEL_ALBANY FALSE)
  ENDIF()
else ()
  # This block is for testing. Set "if (1)" to "if (0)", and then freely mess
  # around with the settings in this block.

  # What to build and test
  set (DOWNLOAD_TRILINOS FALSE)
  set (DOWNLOAD_ALBANY FALSE)
  # See if we can get away with this for speed, at least until we get onto a
  # machine that can support a lengthy nightly.
  set (CLEAN_BUILD FALSE)
  set (BUILD_TRILINOS FALSE)
  set (BUILD_PERIDIGM TRUE)
  set (BUILD_ALB32 FALSE)
  set (BUILD_ALB64 FALSE)
  set (BUILD_TRILINOSCLANG FALSE)
  set (BUILD_ALB64CLANG FALSE)
  set (BUILD_TRILINOSCLANGDBG FALSE)
  set (BUILD_ALB64CLANGDBG FALSE)
  set (BUILD_ALBFUNCTOR FALSE)
  set (BUILD_INTEL_TRILINOS FALSE)
  set (BUILD_INTEL_ALBANY FALSE)
  set (CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?
#  set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
endif ()

set (extra_cxx_flags "")

find_program(UNAME NAMES uname)
macro(getuname name flag)
  exec_program("${UNAME}" ARGS "${flag}" OUTPUT_VARIABLE "${name}")
endmacro(getuname)

getuname(osname -s)
getuname(osrel  -r)
getuname(cpu    -m)

# Begin User inputs:
#set (CTEST_SITE "cee-compute011.sandia.gov" ) # generally the output of hostname
SITE_NAME(CTEST_SITE) # directly set CTEST_SITE to the output of `hostname`
set (CTEST_DASHBOARD_ROOT "$ENV{INSTALL_DIRECTORY}" ) # writable path
set (CTEST_SCRATCH_ROOT "$ENV{SCRATCH_DIRECTORY}" ) # writable path
set (CTEST_SCRIPT_ROOT "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set (CTEST_CMAKE_GENERATOR "Unix Makefiles" ) # What is your compilation apps ?

set (CTEST_PROJECT_NAME "Albany" )
set (CTEST_SOURCE_NAME repos)
#set (CTEST_BUILD_NAME "linux-gcc-${CTEST_BUILD_CONFIGURATION}")
#set (CTEST_BUILD_NAME "${osname}-${osrel}-${CTEST_BUILD_OPTION}-${CTEST_BUILD_CONFIGURATION}")
set (CTEST_BUILD_NAME "${osname}-${osrel}-${CTEST_BUILD_OPTION}")
set (CTEST_BINARY_NAME build)
set (CTEST_INSTALL_NAME test)

if (CTEST_BUILD_CONFIGURATION MATCHES "Debug")
# Runs tests longer if in debug mode
   set (CTEST_TEST_TIMEOUT 4200)
endif ()

set (PREFIX_DIR /projects/albany)
set (INTEL_PREFIX_DIR ${PREFIX_DIR}/intel5.1)
#set (GCC_MPI_DIR /projects/sierra/linux_rh6/SDK/mpi/openmpi/1.10.2-gcc-5.4.0-RHEL6)
#set (GCC_DBG_MPI_DIR /projects/sierra/linux_rh6/SDK/mpi/openmpi/1.10.2-gcc-7.2.0-RHEL6)
set (GCC_MPI_DIR $ENV{MPI_HOME})
set (GCC_DBG_MPI_DIR $ENV{MPI_HOME})

set (BOOST_ROOT /projects/albany)
set (INTEL_BOOST_ROOT ${BOOST_ROOT}/intel5.1)
#set (CLANG_BOOST_ROOT ${BOOST_ROOT}/clang)
set (CLANG_BOOST_ROOT ${BOOST_ROOT}/clang/boost_1_55_0_clang)

SET (INTEL_MPI_DIR $ENV{MPI_HOME})
SET (MPI_BIN_DIR $ENV{MPI_BIN})
SET (MPI_LIB_DIR $ENV{MPI_LIB})

set (CTEST_SOURCE_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_SOURCE_NAME}")
# Build all results in a scratch space
set (CTEST_BINARY_DIRECTORY "${CTEST_SCRATCH_ROOT}/${CTEST_BINARY_NAME}")
# Trilinos, etc installed here
set (CTEST_INSTALL_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_INSTALL_NAME}")

if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}")
  file (MAKE_DIRECTORY "${CTEST_SOURCE_DIRECTORY}")
endif ()
if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}")
  file (MAKE_DIRECTORY "${CTEST_BINARY_DIRECTORY}")
endif ()
if (NOT EXISTS "${CTEST_INSTALL_DIRECTORY}")
  file (MAKE_DIRECTORY "${CTEST_INSTALL_DIRECTORY}")
endif ()

# Clean up storage area for nightly testing results
IF (CLEAN_BUILD)
  IF(EXISTS "${CTEST_BINARY_DIRECTORY}/Testing" )
    FILE(REMOVE_RECURSE "${CTEST_BINARY_DIRECTORY}/Testing")
  ENDIF()
ENDIF()

configure_file (${CTEST_SCRIPT_DIRECTORY}/CTestConfig.cmake
  ${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake COPYONLY)

set (CTEST_NIGHTLY_START_TIME "01:00:00 UTC")
set (CTEST_CMAKE_COMMAND "${PREFIX_DIR}/bin/cmake")
set (CTEST_COMMAND "${PREFIX_DIR}/bin/ctest -D ${CTEST_TEST_TYPE}")
set (CTEST_BUILD_FLAGS "-j16")

set (CTEST_DROP_METHOD "https")

if (CTEST_DROP_METHOD STREQUAL "https")
  set (CTEST_DROP_SITE "sems-cdash-son.sandia.gov")
  set (CTEST_PROJECT_NAME "Albany")
  set (CTEST_DROP_LOCATION "/cdash/submit.php?project=Albany")
  set (CTEST_TRIGGER_SITE "")
  set (CTEST_DROP_SITE_CDASH TRUE)
endif ()

find_program (CTEST_GIT_COMMAND NAMES git)
find_program (CTEST_SVN_COMMAND NAMES svn)

set (Trilinos_REPOSITORY_LOCATION git@github.com:trilinos/Trilinos.git)
set (Albany_REPOSITORY_LOCATION git@github.com:SNLComputation/Albany.git)
#set (Peridigm_REPOSITORY_LOCATION git@github.com:peridigm/peridigm) #ssh://software.sandia.gov/git/peridigm)

#if (CLEAN_BUILD)
#  # Initial cache info
#  set (CACHE_CONTENTS "
#  SITE:STRING=${CTEST_SITE}
#  CMAKE_BUILD_TYPE:STRING=Release
#  CMAKE_GENERATOR:INTERNAL=${CTEST_CMAKE_GENERATOR}
#  BUILD_TESTING:BOOL=OFF
#  PRODUCT_REPO:STRING=${Albany_REPOSITORY_LOCATION}
#  " )
#
##  ctest_empty_binary_directory( "${CTEST_BINARY_DIRECTORY}" )
#  file(WRITE "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt" "${CACHE_CONTENTS}")
#endif ()

if (DOWNLOAD_TRILINOS)
  #
  # Get the internal Trilinos repo
  #

  set (CTEST_CHECKOUT_COMMAND)

  if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/Trilinos")
    execute_process (COMMAND "${CTEST_GIT_COMMAND}" 
      clone --branch develop ${Trilinos_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/Trilinos
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

ENDIF()

IF (DOWNLOAD_ALBANY) 
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
  #if (BUILD_PERIDIGM AND (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/Peridigm"))
  #  execute_process (COMMAND ${CTEST_GIT_COMMAND}
  #    clone ${Peridigm_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/Peridigm
  #    OUTPUT_VARIABLE _out
  #    ERROR_VARIABLE _err
  #    RESULT_VARIABLE HAD_ERROR)
  #  message(STATUS "out: ${_out}")
  #  message(STATUS "err: ${_err}")
  #  message(STATUS "res: ${HAD_ERROR}")
  #  if (HAD_ERROR)
  #    message (FATAL_ERROR "Cannot clone Peridigm repository.")
  #    set (BUILD_PERIDIGM FALSE)
  #  endif ()    
  #endif ()
ENDIF ()

ctest_start(${CTEST_TEST_TYPE})

#
# Send the project structure to CDash
#

if (CTEST_DO_SUBMIT)
  ctest_submit (FILES "${CTEST_SCRIPT_DIRECTORY}/Project.xml"
    RETURN_VALUE HAD_ERROR)

  if (HAD_ERROR)
    message ("Cannot submit Albany Project.xml!")
  endif (HAD_ERROR)
endif (CTEST_DO_SUBMIT)

if (DOWNLOAD_TRILINOS)

  #
  # Update Trilinos
  #

  CTEST_UPDATE(SOURCE "${CTEST_SOURCE_DIRECTORY}/Trilinos" RETURN_VALUE count)
  # assumes that we are already on the desired tracking branch, i.e.,
  # git checkout -b branch --track origin/branch
  message("Found ${count} changed files")

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Update
      RETURN_VALUE  HAD_ERROR
      )

    if (HAD_ERROR)
      message("Cannot update Trilinos!")
    endif ()
  endif ()

  if (count LESS 0)
    message(FATAL_ERROR "Cannot update Trilinos!")
  endif ()


ENDIF()

IF(DOWNLOAD_ALBANY) 
  #
  # Update Albany 
  #

  set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
  CTEST_UPDATE(SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany" RETURN_VALUE count)
  message("Found ${count} changed files")

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Update
      RETURN_VALUE  HAD_ERROR
      )

    if (HAD_ERROR)
      message("Cannot update Albany repository!")
    endif ()
  endif ()

  if (count LESS 0)
    message(FATAL_ERROR "Cannot update Albany!")
  endif ()

  # Peridigm
  #if (BUILD_PERIDIGM)

 #   set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
 #   CTEST_UPDATE (SOURCE "${CTEST_SOURCE_DIRECTORY}/Peridigm" RETURN_VALUE count)
 #   message ("Found ${count} changed files")
 #   if (count LESS 0)
 #     set (BUILD_PERIDIGM FALSE)
 #   endif ()

  #  if (CTEST_DO_SUBMIT)
  #    ctest_submit (PARTS Update RETURN_VALUE HAD_ERROR)
  #  endif ()

   # message ("After downloading, BUILD_PERIDIGM = ${BUILD_PERIDIGM}")
  #endif ()

endif ()

#
# Set the common Trilinos config options
#

set (COMMON_CONFIGURE_OPTIONS
  "-Wno-dev"
  #
  "-DTrilinos_ENABLE_ThyraTpetraAdapters:BOOL=ON"
  "-DTrilinos_ENABLE_ThyraEpetraAdapters:BOOL=ON"
  "-DTrilinos_ENABLE_Ifpack2:BOOL=ON"
  "-DTrilinos_ENABLE_Amesos2:BOOL=ON"
  "-DTrilinos_ENABLE_Zoltan2:BOOL=ON"
  "-DTrilinos_ENABLE_MueLu:BOOL=ON"
  "-DTrilinos_CXX11_FLAGS:STRING='-std=c++11'"
#
  "-DTrilinos_WARNINGS_AS_ERRORS_FLAGS:BOOL=OFF"
  "-DTrilinos_ENABLE_STRONG_C_COMPILE_WARNINGS:BOOL=OFF"
  "-DTrilinos_ENABLE_STRONG_CXX_COMPILE_WARNINGS:BOOL=OFF"
  "-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON"
  "-DTrilinos_SHOW_DEPRECATED_WARNINGS:BOOL=OFF"
  #
  "-DZoltan_ENABLE_ULONG_IDS:BOOL=ON"
  "-DMDS_ID_TYPE:STRING='long int'"
  "-DZOLTAN_BUILD_ZFDRIVE:BOOL=OFF"
  "-DTpetra_INST_INT_LONG_LONG:BOOL=ON"
  "-DTpetra_INST_INT_LONG:BOOL=OFF"
  "-DTpetra_INST_INT_INT:BOOL=OFF"
  "-DXpetra_ENABLE_Epetra=OFF"
  "-DMueLu_ENABLE_Epetra=OFF"
  "-DBelos_ENABLE_Epetra=OFF"
  "-DTpetra_INST_DOUBLE:BOOL=ON"
  "-DTpetra_INST_FLOAT:BOOL=OFF"
  "-DTpetra_INST_COMPLEX_FLOAT:BOOL=OFF"
  "-DTpetra_INST_COMPLEX_DOUBLE:BOOL=OFF"
  "-DTpetra_INST_INT_UNSIGNED:BOOL=OFF"
  "-DTpetra_INST_INT_UNSIGNED_LONG:BOOL=OFF"
  "-DTeuchos_ENABLE_COMPLEX:BOOL=OFF"
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
  #
  #
  "-DTPL_BLAS_LIBRARIES:STRING='-L$ENV{LIBRARY_PATH} -L$ENV{MKLHOME}/../compiler/lib/intel64 -lmkl_intel_lp64 -lmkl_blas95_lp64 -lmkl_core -lmkl_sequential -lmkl_core -lirc -limf -lsvml -lintlc'"
  "-DTPL_LAPACK_LIBRARIES:STRING='-L$ENV{LIBRARY_PATH} -lmkl_lapack95_lp64'"
  #
  "-DTrilinos_ENABLE_TESTS:BOOL=OFF"
  "-DTrilinos_ENABLE_EXAMPLES:BOOL=OFF"
  "-DMueLu_ENABLE_Tutorial:BOOL=OFF"
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
  "-DAnasazi_ENABLE_RBGen:BOOL=ON"
  "-DTrilinos_ENABLE_TpetraTSQR:BOOL=ON"
  "-DTpetraCore_ENABLE_TSQR:BOOL=ON"
  "-DBelos_ENABLE_TSQR:BOOL=ON"
  "-DTrilinos_ENABLE_Belos:BOOL=ON"
  "-DTrilinos_ENABLE_ML:BOOL=ON"
  "-DTrilinos_ENABLE_Phalanx:BOOL=ON"
  "-DTrilinos_ENABLE_Intrepid2:BOOL=ON"
  "-DTrilinos_ENABLE_ROL:BOOL=ON"
  "-DTrilinos_ENABLE_MiniTensor:BOOL=ON"
  "-DTrilinos_ENABLE_NOX:BOOL=ON"
  "-DTrilinos_ENABLE_Stratimikos:BOOL=ON"
  "-DTrilinos_ENABLE_Thyra:BOOL=ON"
  "-DTrilinos_ENABLE_Rythmos:BOOL=ON"
  "-DTrilinos_ENABLE_Stokhos:BOOL=OFF"
  "-DTrilinos_ENABLE_Isorropia:BOOL=ON"
  "-DTrilinos_ENABLE_Piro:BOOL=ON"
  "-DTrilinos_ENABLE_Teko:BOOL=ON"
  "-DTrilinos_ENABLE_Zoltan:BOOL=ON"
  #
  "-DTrilinos_ENABLE_FEI:BOOL=OFF"
  #
  "-DPhalanx_ENABLE_TEUCHOS_TIME_MONITOR:BOOL=ON"
  "-DStratimikos_ENABLE_TEUCHOS_TIME_MONITOR:BOOL=ON"
  #
  "-DTrilinos_ENABLE_SEACAS:BOOL=ON"
  "-DTrilinos_ENABLE_Pamgen:BOOL=ON"
  "-DTrilinos_ENABLE_PanzerExprEval:BOOL=ON"
  "-DTrilinos_ENABLE_PyTrilinos:BOOL=OFF"
  #
  #"-DTrilinos_ENABLE_STK:BOOL=ON"
  #"-DTrilinos_ENABLE_STKExprEval:BOOL=ON"
  "-DTrilinos_ENABLE_SEACASIoss:BOOL=ON"
  "-DTrilinos_ENABLE_SEACASExodus:BOOL=ON"
  #"-DTrilinos_ENABLE_STKUtil:BOOL=ON"
  #"-DTrilinos_ENABLE_STKTopology:BOOL=ON"
  "-DTrilinos_ENABLE_STKMesh:BOOL=ON"
  "-DTrilinos_ENABLE_STKIO:BOOL=ON"
  "-DTrilinos_ENABLE_STKExprEval:BOOL=ON"
  #"-DTrilinos_ENABLE_STKExp:BOOL=OFF"
  #"-DTrilinos_ENABLE_STKSearch:BOOL=ON"
  #"-DTrilinos_ENABLE_STKSearchUtil:BOOL=ON"
  #"-DTrilinos_ENABLE_STKTransfer:BOOL=ON"
  #"-DTrilinos_ENABLE_STKUnit_tests:BOOL=OFF"
  #"-DTrilinos_ENABLE_STKDoc_tests:BOOL=OFF"
  #
  "-DTrilinos_ENABLE_Kokkos:BOOL=ON"
  "-DTrilinos_ENABLE_KokkosCore:BOOL=ON"
  "-DPhalanx_INDEX_SIZE_TYPE:STRING=KOKKOS"
  "-DKokkos_ENABLE_SERIAL:BOOL=ON"
  "-DKokkos_ENABLE_OPENMP:BOOL=OFF"
  "-DKokkos_ENABLE_PTHREAD:BOOL=OFF"
  #
  "-DTrilinos_ENABLE_Tempus:BOOL=ON"
  #
  "-DSTK_HIDE_DEPRECATED_CODE:BOOL=OFF"
  "-DTpetra_ENABLE_DEPRECATED_CODE:BOOL=OFF"
  "-DXpetra_ENABLE_DEPRECATED_CODE:BOOL=OFF"
  "-DUSE_NEW_POLICY_CMP0060=NEW"
  )

if (CTEST_BUILD_CONFIGURATION MATCHES "Debug")
   set (COMMON_CONFIGURE_OPTIONS ${COMMON_CONFIGURE_OPTIONS}
     "-DDART_TESTING_TIMEOUT:STRING=4200"
   )
else ()
   set (COMMON_CONFIGURE_OPTIONS ${COMMON_CONFIGURE_OPTIONS}
     "-DDART_TESTING_TIMEOUT:STRING=600"
   )
endif ()

INCLUDE(${CTEST_SCRIPT_DIRECTORY}/trilinos_macro.cmake)

if (BUILD_TRILINOS OR BUILD_TRILINOSDBG OR BUILD_TRILINOSCLANG OR BUILD_TRILINOSCLANGDBG)

  if (BUILD_TRILINOS) 
    set(INSTALL_LOCATION "${CTEST_INSTALL_DIRECTORY}/TrilinosInstall")
    set(BTYPE "RELEASE") 
    set(CCFLAGS "-O3 -march=native -DNDEBUG -Wno-inconsistent-missing-override")
    set(CFLAGS "-O3 -march=native -DNDEBUG")
    set(FFLAGS "-O3 -march=native -DNDEBUG -Wa,-q")
  endif(BUILD_TRILINOS)
  if (BUILD_TRILINOSCLANG) 
    set(INSTALL_LOCATION "${CTEST_INSTALL_DIRECTORY}/TrilinosInstallC11")
    set(BTYPE "RELEASE") 
    set(CCFLAGS "-O3 -march=native -DNDEBUG=1")
    set(CFLAGS "-O3 -march=native -DNDEBUG=1")
    set(FFLAGS "-O3 -march=native -DNDEBUG=1 -Wa,-q")
  endif (BUILD_TRILINOSCLANG) 
  if (BUILD_TRILINOSDBG) 
    set(INSTALL_LOCATION "${CTEST_INSTALL_DIRECTORY}/TrilinosDbg")
    set(BTYPE "DEBUG") 
    set(CCFLAGS "-g -O0 -Wno-inconsistent-missing-override") 
    set(CFLAGS "-g -O0")
    set(FFLAGS "-g -O0 -Wa,-q") 
  endif(BUILD_TRILINOSDBG)
  if (BUILD_TRILINOSCLANGDBG) 
    set(INSTALL_LOCATION "${CTEST_INSTALL_DIRECTORY}/TrilinosInstallC11Dbg")
    set(BTYPE "DEBUG")
    set(CCFLAGS "-g -O0")
    set(CFLAGS "-g -O0")
    set(FFLAGS "-g -O0 -Wa,-q")
  endif (BUILD_TRILINOSCLANGDBG) 
  if (BUILD_TRILINOS OR BUILD_TRILINOSDBG)  
    set(BOOST_DIR "/projects/albany/gcc-9.1.0")
    set(LIB_DIR "/projects/albany/gcc-9.1.0")
    set(GCC_LIB_DIR "/projects/albany/gcc-9.1.0")
    set(MPI_BASE_DIR "/projects/albany/gcc-9.1.0")
    set(NETCDF "/projects/albany/gcc-9.1.0") 
    set(HDFDIR "/projects/albany/gcc-9.1.0")
    set(PARMETISDIR "/projects/albany/gcc-9.1.0")
    set(MKL_PATH "/sierra/sntools/SDK/compilers/intel/composer_xe_2019.5.281") 
    set(SUPERLUDIR "/projects/albany/gcc-9.1.0/SuperLU_4.3") 
    set(LABLAS_LIBRARIES "-L${MKL_PATH}/lib/intel64 -Wl,--start-group ${MKL_PATH}/mkl/lib/intel64/libmkl_intel_lp64.a ${MKL_PATH}/mkl/lib/intel64/libmkl_core.a ${MKL_PATH}/mkl/lib/intel64/libmkl_sequential.a -Wl,--end-group") 
  endif (BUILD_TRILINOS OR BUILD_TRILINOSDBG)  
  if (BUILD_TRILINOSCLANG OR BUILD_TRILINOSCLANGDBG) 
    set(BOOST_DIR "/projects/albany/clang-9.0.1")
    set(LIB_DIR "/projects/albany/clang-9.0.1")
    set(GCC_LIB_DIR "/projects/albany/gcc-9.1.0")
    set(MPI_BASE_DIR "/projects/albany/clang-9.0.1")
    set(NETCDF "/projects/albany/clang-9.0.1")
    set(HDFDIR "/projects/albany/clang-9.0.1")
    set(PARMETISDIR "/projects/albany/clang-9.0.1")
    set(MKL_PATH "/sierra/sntools/SDK/compilers/intel/composer_xe_2019.5.281")
    set(SUPERLUDIR "/projects/albany/clang-9.0.1/SuperLU_4.3")
    set(LABLAS_LIBRARIES "-L${MKL_PATH}/lib/intel64 -Wl,--start-group ${MKL_PATH}/mkl/lib/intel64/libmkl_intel_lp64.a ${MKL_PATH}/mkl/lib/intel64/libmkl_core.a ${MKL_PATH}/mkl/lib/intel64/libmkl_sequential.a -Wl,--end-group")
  endif (BUILD_TRILINOSCLANG OR BUILD_TRILINOSCLANGDBG) 
  set (CONF_OPTS
    "-DCMAKE_BUILD_TYPE:STRING=${BTYPE}"
    "-DCMAKE_CXX_COMPILER:STRING=${MPI_BASE_DIR}/bin/mpicxx"
    "-DCMAKE_CXX_FLAGS:STRING=${CCFLAGS}"
    "-DCMAKE_C_COMPILER:STRING=${MPI_BASE_DIR}/bin/mpicc"
    "-DCMAKE_C_FLAGS:STRING=${CFLAGS}"
    "-DCMAKE_Fortran_COMPILER:STRING=${MPI_BASE_DIR}/bin/mpif90"
    "-DCMAKE_Fortran_FLAGS:STRING=${FFLAGS}"
    "-DTrilinos_SHOW_DEPRECATED_WARNINGS:BOOL=OFF"
    "-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON"
    "-DTrilinos_ENABLE_OpenMP:BOOL=OFF"
    "-DTPL_ENABLE_MPI:BOOL=ON"
    "-DMPI_BASE_DIR:PATH=${MPI_BASE_DIR}"
if (BUILD_TRILINOS OR BUILD_TRILINOSDBG) 
    "-DCMAKE_MACOSX_RPATH:BOOL=ON"
endif (BUILD_TRILINOS OR BUILD_TRILINOSDBG) 
    "-DCMAKE_INSTALL_RPATH:PATH=${MPI_BASE_DIR}/lib"
    "-DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_LOCATION}"
    #
    "-DTrilinos_ENABLE_ALL_PACKAGES:BOOL=OFF"
    "-DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF"
    "-DTrilinos_ENABLE_SECONDARY_TESTED_CODE:BOOL=ON"
    "-DTrilinos_ENABLE_TESTS:BOOL=OFF"
    "-DTrilinos_ENABLE_EXAMPLES:BOOL=OFF"
    "-DTrilinos_ASSERT_MISSING_PACKAGES:BOOL=OFF"
    #
    "-DTrilinos_ENABLE_AztecOO:BOOL=ON"
    "-DTrilinos_ENABLE_Amesos:BOOL=ON"
    "-DTrilinos_ENABLE_Amesos2:BOOL=ON"
    "-DAmesos2_ENABLE_KLU2:BOOL=ON"
    "-DTrilinos_ENABLE_Anasazi:BOOL=ON"
    "-DTrilinos_ENABLE_Belos:BOOL=ON"
    "-DTrilinos_ENABLE_Epetra:BOOL=ON"
    "-DTrilinos_ENABLE_EpetraExt:BOOL=ON"
    "-DTrilinos_ENABLE_Intrepid2:BOOL=ON"
    #
    "-DTrilinos_ENABLE_ROL:BOOL=ON"
    "-DTrilinos_ENABLE_MiniTensor:BOOL=ON"
    "-DTrilinos_ENABLE_ML:BOOL=ON"
    "-DTrilinos_ENABLE_MueLu:BOOL=ON"
    "-DMueLu_ENABLE_Tutorial:BOOL=OFF"
    "-DTrilinos_ENABLE_Moertel:BOOL=OFF"
    "-DTrilinos_ENABLE_NOX:BOOL=ON"
    "-DTrilinos_ENABLE_Rythmos:BOOL=ON"
    "-DTrilinos_ENABLE_Sacado:BOOL=ON"
    "-DTrilinos_ENABLE_SCOREC:BOOL=OFF"
    "-DTrilinos_ENABLE_SEACAS:BOOL=ON"
    "-DTrilinos_ENABLE_Shards:BOOL=ON"
    "-DTrilinos_ENABLE_Stokhos:BOOL=OFF"
    "-DTrilinos_ENABLE_STK:BOOL=ON"
    "-DTrilinos_ENABLE_STKSearch:BOOL=ON"
    "-DTrilinos_ENABLE_Stratimikos:BOOL=ON"
    "-DTrilinos_ENABLE_Ifpack:BOOL=ON"
    "-DTrilinos_ENABLE_Isorropia:BOOL=ON"
    "-DTrilinos_ENABLE_Pamgen:BOOL=ON"
    "-DTrilinos_ENABLE_Phalanx:BOOL=ON"
    "-DTrilinos_ENABLE_Piro:BOOL=ON"
    "-DTrilinos_ENABLE_Teuchos:BOOL=ON"
    "-DTrilinos_ENABLE_Teko:BOOL=ON"
    "-DTrilinos_ENABLE_Tempus:BOOL=ON"
    "-DTrilinos_ENABLE_Thyra:BOOL=ON"
    "-DTrilinos_ENABLE_Zoltan:BOOL=ON"
    #
    "-DRythmos_ENABLE_DEBUG:BOOL=ON"
    "-DTpetra_INST_INT_LONG_LONG:BOOL=ON"
    "-DTpetra_INST_INT_LONG:BOOL=OFF"
    "-DTpetra_INST_INT_INT:BOOL=OFF"
    "-DTpetra_INST_DOUBLE:BOOL=ON"
    "-DTpetra_INST_FLOAT:BOOL=OFF"
    "-DTpetra_INST_COMPLEX_FLOAT:BOOL=OFF"
    "-DTpetra_INST_COMPLEX_DOUBLE:BOOL=OFF"
    "-DTpetra_INST_INT_UNSIGNED:BOOL=OFF"
    "-DTpetra_INST_INT_UNSIGNED_LONG:BOOL=OFF"
    "-DZoltan_ENABLE_ULONG_IDS:BOOL=ON"
    "-DTeuchos_ENABLE_COMPLEX:BOOL=OFF"
    "-DZOLTAN_BUILD_ZFDRIVE:BOOL=OFF"
    "-DPhalanx_INDEX_SIZE_TYPE:STRING='KOKKOS'"
    "-DKokkos_ENABLE_SERIAL:BOOL=ON"
    "-DKokkos_ENABLE_OPENMP:BOOL=OFF"
    "-DKokkos_ENABLE_PTHREAD:BOOL=OFF"
    #
    "-DSEACAS_ENABLE_SEACASSVDI:BOOL=OFF"
    "-DTrilinos_ENABLE_SEACASFastq:BOOL=OFF"
    "-DTrilinos_ENABLE_SEACASBlot:BOOL=OFF"
    "-DTrilinos_ENABLE_SEACASPLT:BOOL=OFF"
    "-DTPL_ENABLE_X11:BOOL=OFF"
    "-DTPL_ENABLE_Matio:BOOL=OFF"
    #
    "-D TPL_ENABLE_MPI:BOOL=ON"
    "-DMPI_BASE_DIR:PATH=${MPI_BASE_DIR}"
    "-DMPI_EXEC=${MPI_BASE_DIR}/bin/mpiexec"
    #
    "-DTPL_ENABLE_Pthread:BOOL=OFF"
    #
    "-DTPL_ENABLE_Boost:BOOL=ON"
    "-DTPL_ENABLE_BoostLib:BOOL=ON"
    "-DTPL_ENABLE_BoostAlbLib:BOOL=ON"
    "-DBoost_INCLUDE_DIRS:PATH=${BOOST_DIR}/include"
    "-DBoost_LIBRARY_DIRS:PATH=${BOOST_DIR}/lib"
    "-DBoostLib_INCLUDE_DIRS:PATH=${BOOST_DIR}/include"
    "-DBoostLib_LIBRARY_DIRS:PATH=${BOOST_DIR}/lib"
    "-DBoostAlbLib_INCLUDE_DIRS:PATH=${BOOST_DIR}/include"
    "-DBoostAlbLib_LIBRARY_DIRS:PATH=${BOOST_DIR}/lib"
    #
    "-DTPL_Netcdf_PARALLEL:BOOL=ON"
    "-DTPL_ENABLE_Netcdf:STRING=ON"
    "-DNetcdf_INCLUDE_DIRS:PATH=${NETCDF}/include"
    "-DNetcdf_LIBRARY_DIRS:PATH=${NETCDF}/lib"
    "-DTPL_ENABLE_Pnetcdf:BOOL=ON"
    "-DPnetcdf_INCLUDE_DIRS:PATH=${NETCDF}/include"
    "-DPnetcdf_LIBRARY_DIRS=${NETCDF}/lib"
    #
    "-DTPL_ENABLE_HDF5:STRING=ON"
    "-DHDF5_INCLUDE_DIRS:PATH=${HDFDIR}/include"
    "-DHDF5_LIBRARY_DIRS:PATH=${HDFDIR}/lib"
    #
    "-DTPL_ENABLE_Zlib:STRING=ON"
    "-DZlib_INCLUDE_DIRS:PATH=${LIB_DIR}/include"
    "-DZlib_LIBRARY_DIRS:PATH=${LIB_DIR}/lib"
    #
    "-DTPL_ENABLE_SuperLU:STRING=ON"
    "-DSuperLU_INCLUDE_DIRS:STRING=${SUPERLUDIR}/include"
    "-DSuperLU_LIBRARY_DIRS:STRING=${SUPERLUDIR}/lib"
    #
    "-DTPL_ENABLE_BLAS:STRING=ON"
    "-DTPL_ENABLE_LAPACK:STRING=ON"
    "-DTPL_BLAS_LIBRARIES:STRING=${LABLAS_LIBRARIES}"
    "-DTPL_LAPACK_LIBRARIES:STRING=${LABLAS_LIBRARIES}"
    #
    "-DTPL_ENABLE_ParMETIS:STRING=OFF"
    #
    "-DTrilinos_EXTRA_LINK_FLAGS:STRING='-L${LIB_DIR}/lib -L${LIB_DIR}/lib64 -lnetcdf -lpnetcdf -lhdf5_hl -lhdf5 -lz -lgfortran -Wl,-rpath,${GCC_LIB_DIR}/lib:${GCC_LIB_DIR}/lib64'"
    #
    "-DTrilinos_ENABLE_TriKota:BOOL=OFF"
    "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON"
    "-DCMAKE_INSTALL_RPATH:STRING='/projects/albany/gcc-9.1.0/lib'"
  )

  # First argument is the string of the configure options, second is the dashboard target (a name in a string)
  do_trilinos("${CONF_OPTS}" "Trilinos" "${INSTALL_LOCATION}")

endif (BUILD_TRILINOS OR BUILD_TRILINOSDBG OR BUILD_TRILINOSCLANG OR BUILD_TRILINOSCLANGDBG)

if (BUILD_PERIDIGM)
  INCLUDE(${CTEST_SCRIPT_DIRECTORY}/peridigm_macro.cmake)
  do_peridigm()
endif (BUILD_PERIDIGM)

INCLUDE(${CTEST_SCRIPT_DIRECTORY}/albany_macro.cmake)

if (BUILD_ALB32)

  set (CONF_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${CTEST_INSTALL_DIRECTORY}/TrilinosInstall"
    "-DENABLE_LANDICE:BOOL=ON"
    "-DENABLE_UNIT_TESTS:BOOL=ON"
    "-DENABLE_STRONG_FPE_CHECK:BOOL=ON"
    )
  # First argument is the string of the configure options, second is the dashboard target (a name in a string)
  do_albany("${CONF_OPTIONS}" "Albany32Bit")

endif (BUILD_ALB32)

#
# Configure the Albany build using GO = long
#

if (BUILD_ALB64)

  set (CONF_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${CTEST_INSTALL_DIRECTORY}/TrilinosInstall"
    "-DENABLE_64BIT_INT:BOOL=ON"
#    "-DENABLE_ALBANY_EPETRA:BOOL=OFF"
    "-DENABLE_LANDICE:BOOL=ON"
    "-DENABLE_UNIT_TESTS:BOOL=ON"
    "-DENABLE_STRONG_FPE_CHECK:BOOL=ON"
    )

  # First argument is the string of the configure options, second is the dashboard target (a name in a string)
  do_albany("${CONF_OPTIONS}" "Albany64Bit")

endif (BUILD_ALB64)

#
# Configure the Albany Clang build using GO = long
#

if (BUILD_ALB64CLANG)

  set (CONF_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${CTEST_INSTALL_DIRECTORY}/TrilinosInstallC11"
    "-DENABLE_64BIT_INT:BOOL=ON"
# Run even the epetra tests
#    "-DENABLE_ALBANY_EPETRA:BOOL=OFF"
    "-DENABLE_LANDICE:BOOL=ON"
    "-DENABLE_UNIT_TESTS:BOOL=ON"
    "-DENABLE_STRONG_FPE_CHECK:BOOL=OFF"
    "-DENABLE_MESH_DEPENDS_ON_SOLUTION:BOOL=ON"
    )

  # First argument is the string of the configure options, second is the dashboard target (a name in a string)
  do_albany("${CONF_OPTIONS}" "Albany64BitClang")

endif (BUILD_ALB64CLANG)

if (BUILD_ALB64CLANGDBG)

  set (CONF_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${CTEST_INSTALL_DIRECTORY}/TrilinosInstallC11Dbg"
    "-DENABLE_64BIT_INT:BOOL=ON"
# Run even the epetra tests
#    "-DENABLE_ALBANY_EPETRA:BOOL=OFF"
    "-DENABLE_LANDICE:BOOL=ON"
    "-DENABLE_UNIT_TESTS:BOOL=ON"
    "-DENABLE_STRONG_FPE_CHECK:BOOL=ON"
    "-DENABLE_MESH_DEPENDS_ON_PARAMETERS:BOOL=ON"
    "-DCMAKE_BUILD_TYPE:STRING=DEBUG"
    )

  # First argument is the string of the configure options, second is the dashboard target (a name in a string)
  do_albany("${CONF_OPTIONS}" "Albany64BitClangDbg")

endif (BUILD_ALB64CLANGDBG)

#
# Configure the Albany build using GO = long
#

if (BUILD_ALB64DBG)

  set (CONF_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${CTEST_INSTALL_DIRECTORY}/TrilinosDbg"
    "-DENABLE_64BIT_INT:BOOL=ON"
    "-DENABLE_ALBANY_EPETRA:BOOL=OFF"
    "-DENABLE_LANDICE:BOOL=ON"
    "-DENABLE_UNIT_TESTS:BOOL=ON"
    "-DENABLE_STRONG_FPE_CHECK:BOOL=ON"
    )

  # First argument is the string of the configure options, second is the dashboard target (a name in a string)
  do_albany("${CONF_OPTIONS}" "Albany64BitDbg")

endif (BUILD_ALB64DBG)


if (BUILD_ALBFUNCTOR)

  set (CONF_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${CTEST_INSTALL_DIRECTORY}/TrilinosInstall"
    "-DENABLE_LANDICE:BOOL=ON"
    "-DENABLE_UNIT_TESTS:BOOL=ON"
    "-DENABLE_64BIT_INT:BOOL=ON"
    "-DENABLE_DEMO_PDES:BOOL=ON"
    "-DENABLE_KOKKOS_UNDER_DEVELOPMENT:BOOL=ON"
    "-DENABLE_STRONG_FPE_CHECK:BOOL=ON"
    )

  # First argument is the string of the configure options, second is the dashboard target (a name in a string)
  do_albany("${CONF_OPTIONS}" "AlbanyFunctorDev")

endif (BUILD_ALBFUNCTOR)

if (BUILD_INTEL_TRILINOS OR BUILD_INTEL_ALBANY)
  INCLUDE(${CTEST_SCRIPT_DIRECTORY}/intel_macro.cmake)

  # First argument is the string of the configure options, second is the dashboard target (a name in a string)
   do_intel("${COMMON_CONFIGURE_OPTIONS}" "TrilinosIntel")

endif (BUILD_INTEL_TRILINOS OR BUILD_INTEL_ALBANY)
