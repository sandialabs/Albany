cmake_minimum_required(VERSION 2.8)

SET(CTEST_DO_SUBMIT ON)
SET(CTEST_TEST_TYPE Nightly)

#SET(CTEST_DO_SUBMIT OFF)
#SET(CTEST_TEST_TYPE Experimental)

# What to build and test
SET(BUILD_ALB32 TRUE)
SET(BUILD_ALB64 TRUE)
SET(BUILD_ALB64CLANG11 TRUE)
SET(DOWNLOAD TRUE)
SET(BUILD_TRILINOS TRUE)
SET(BUILD_TRILINOSCLANG11 TRUE)
SET(CLEAN_BUILD TRUE)

# Begin User inputs:
set( CTEST_SITE             "cee-compute011.sandia.gov" ) # generally the output of hostname
set( CTEST_DASHBOARD_ROOT   "$ENV{TEST_DIRECTORY}" ) # writable path
set( CTEST_SCRIPT_DIRECTORY   "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set( CTEST_CMAKE_GENERATOR  "Unix Makefiles" ) # What is your compilation apps ?
set( CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?

set(INITIAL_LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})

set( CTEST_PROJECT_NAME         "Albany" )
set( CTEST_SOURCE_NAME          repos)
set( CTEST_BUILD_NAME           "linux-gcc-${CTEST_BUILD_CONFIGURATION}")
set( CTEST_BINARY_NAME          build)

SET(PREFIX_DIR /projects/albany)
SET(GCC_MPI_DIR /sierra/sntools/SDK/mpi/openmpi/1.6.4-gcc-4.8.2-RHEL6)
SET(INTEL_DIR /sierra/sntools/SDK/compilers/intel/composer_xe_2015.1.133)


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
SET (CTEST_BUILD_FLAGS "-j16")

SET(CTEST_DROP_METHOD "http")

IF (CTEST_DROP_METHOD STREQUAL "http")
  SET(CTEST_DROP_SITE "cdash.sandia.gov")
  SET(CTEST_PROJECT_NAME "Albany")
  SET(CTEST_DROP_LOCATION "/CDash-2-3-0/submit.php?project=Albany")
  SET(CTEST_TRIGGER_SITE "")
  SET(CTEST_DROP_SITE_CDASH TRUE)
ENDIF()

find_program(CTEST_GIT_COMMAND NAMES git)
find_program(CTEST_SVN_COMMAND NAMES svn)

# Point at the public Repo
SET(Trilinos_REPOSITORY_LOCATION ambradl@software.sandia.gov:/git/Trilinos)
SET(SCOREC_REPOSITORY_LOCATION git@github.com:SCOREC/core.git)
SET(Albany_REPOSITORY_LOCATION git@github.com:gahansen/Albany.git)

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
#
# Get the publicTrilinos repo
#
#########################################################################################################

set(CTEST_CHECKOUT_COMMAND)

if(NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/Trilinos")
  EXECUTE_PROCESS(COMMAND "${CTEST_GIT_COMMAND}" 
    clone ${Trilinos_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/Trilinos
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

#
# Get the SCOREC repo
#
###########################################################################################################

if(NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/Trilinos/SCOREC")
#  EXECUTE_PROCESS(COMMAND "${CTEST_SVN_COMMAND}" 
#    checkout ${SCOREC_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/Trilinos/SCOREC
#    OUTPUT_VARIABLE _out
#    ERROR_VARIABLE _err
#    RESULT_VARIABLE HAD_ERROR)
  EXECUTE_PROCESS(COMMAND "${CTEST_GIT_COMMAND}" 
    clone ${SCOREC_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/Trilinos/SCOREC
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
    RESULT_VARIABLE HAD_ERROR)
  
   message(STATUS "out: ${_out}")
   message(STATUS "err: ${_err}")
   message(STATUS "res: ${HAD_ERROR}")
   if(HAD_ERROR)
	message(FATAL_ERROR "Cannot checkout SCOREC repository!")
   endif()
endif()

#
# Get Albany
#
##########################################################################################################

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

#
# Send the project structure to CDash
#
##############################################################################################################

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(FILES "${CTEST_SCRIPT_DIRECTORY}/Project.xml"
               RETURN_VALUE  HAD_ERROR
  )

  if(HAD_ERROR)
	message(FATAL_ERROR "Cannot submit Albany Project.xml!")
  endif()
ENDIF()

IF(DOWNLOAD)

#
# Update Trilinos
#
###########################################################################################################

SET_PROPERTY (GLOBAL PROPERTY SubProject Trilinos)
SET_PROPERTY (GLOBAL PROPERTY Label Trilinos)

ctest_update(SOURCE "${CTEST_SOURCE_DIRECTORY}/Trilinos" RETURN_VALUE count)
message("Found ${count} changed files")

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(PARTS Update
               RETURN_VALUE  HAD_ERROR
  )

  if(HAD_ERROR)
	message(FATAL_ERROR "Cannot update Trilinos!")
  endif()
ENDIF()

IF(count LESS 0)
	message(FATAL_ERROR "Cannot update Trilinos!")
endif()


#
# Update the SCOREC repo
#
###########################################################################################################

SET_PROPERTY (GLOBAL PROPERTY SubProject SCOREC)
SET_PROPERTY (GLOBAL PROPERTY Label SCOREC)

#set(CTEST_UPDATE_COMMAND "${CTEST_SVN_COMMAND}")
set(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
ctest_update(SOURCE "${CTEST_SOURCE_DIRECTORY}/Trilinos/SCOREC" RETURN_VALUE count)
message("Found ${count} changed files")

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(PARTS Update
               RETURN_VALUE  HAD_ERROR
  )

  if(HAD_ERROR)
	message(FATAL_ERROR "Cannot update SCOREC!")
  endif()
ENDIF()

IF(count LESS 0)
	message(FATAL_ERROR "Cannot update Trilinos!")
endif()

#
# Update Albany 
#
##############################################################################################################

SET_PROPERTY (GLOBAL PROPERTY SubProject Albany32Bit)
SET_PROPERTY (GLOBAL PROPERTY Label Albany32Bit)

set(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
CTEST_UPDATE(SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany" RETURN_VALUE count)
message("Found ${count} changed files")

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(PARTS Update
               RETURN_VALUE  HAD_ERROR
  )

  if(HAD_ERROR)
	message(FATAL_ERROR "Cannot update Albany repository!")
  endif()
ENDIF()

IF(count LESS 0)
	message(FATAL_ERROR "Cannot update Albany repository!")
endif()

ENDIF()

#
# Set the common Trilinos config options
#
#######################################################################################################################

SET(COMMON_CONFIGURE_OPTIONS
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
  "-DTPL_ENABLE_Netcdf:STRING=ON"
  "-DNetcdf_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DNetcdf_LIBRARY_DIRS:PATH=${PREFIX_DIR}/lib"
#
  "-DTPL_ENABLE_HDF5:STRING=ON"
  "-DHDF5_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DHDF5_LIBRARY_DIRS:PATH=${PREFIX_DIR}/lib"
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

IF(BUILD_TRILINOS)

#
# Configure the Trilinos/SCOREC build
#
###############################################################################################################

SET_PROPERTY (GLOBAL PROPERTY SubProject Trilinos)
SET_PROPERTY (GLOBAL PROPERTY Label Trilinos)

SET(CONFIGURE_OPTIONS
  "-DTPL_ENABLE_MPI:BOOL=ON"
  "-DMPI_BASE_DIR:PATH=${GCC_MPI_DIR}"
  "-DCMAKE_CXX_FLAGS:STRING='-O3 -march=native -w -DNDEBUG'"
  "-DCMAKE_C_FLAGS:STRING='-O3 -march=native -w -DNDEBUG'"
  "-DCMAKE_Fortran_FLAGS:STRING='-O3 -march=native -w -DNDEBUG'"
  "-DTrilinos_EXTRA_REPOSITORIES:STRING=SCOREC"
  "-DTrilinos_ENABLE_SCOREC:BOOL=ON"
  "-DSCOREC_DISABLE_STRONG_WARNINGS:BOOL=ON"
  "-DTrilinos_EXTRA_LINK_FLAGS='-L${PREFIX_DIR}/lib -lhdf5_hl -lhdf5 -lz -lm'"
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

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(PARTS Configure
               RETURN_VALUE  S_HAD_ERROR
  )

  if(S_HAD_ERROR)
	message(FATAL_ERROR "Cannot submit Trilinos/SCOREC configure results!")
  endif()
ENDIF()

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot configure Trilinos/SCOREC build!")
endif()

#
# SCOREC tools build inside Trilinos
#
# Note that we do a trick here, and just build the SCOREC_libs target, as we build SCOREC as a Trilinos packages
# and its not possible to do that independent of Trilinos. So, while this builds most of SCOREC, other Trilinos 
# capabilities are also built here.
#
#########################################################################################################################

SET_PROPERTY (GLOBAL PROPERTY SubProject SCOREC)
SET_PROPERTY (GLOBAL PROPERTY Label SCOREC)
SET(CTEST_BUILD_TARGET "SCOREC_libs")

MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

CTEST_BUILD(
          BUILD "${CTEST_BINARY_DIRECTORY}/TriBuild"
          RETURN_VALUE  HAD_ERROR
          NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
)

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(PARTS Build
               RETURN_VALUE  S_HAD_ERROR
  )

  if(S_HAD_ERROR)
	message(FATAL_ERROR "Cannot submit SCOREC build results!")
  endif()
ENDIF()

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot build SCOREC!")
endif()

if(BUILD_LIBS_NUM_ERRORS GREATER 0)
	message(FATAL_ERROR "Encountered build errors in SCOREC build. Exiting!")
endif()

#
# Trilinos
#
# Build the rest of Trilinos and install everything
#
########################################################################################################################

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

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(PARTS Build
               RETURN_VALUE  S_HAD_ERROR
  )

  if(S_HAD_ERROR)
	message(FATAL_ERROR "Cannot submit Trilinos/SCOREC build results!")
  endif()

ENDIF()

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot build Trilinos!")
endif()

if(BUILD_LIBS_NUM_ERRORS GREATER 0)
	message(FATAL_ERROR "Encountered build errors in Trilinos build. Exiting!")
endif()

ENDIF()

IF (BUILD_ALB32)

# Configure the Albany 32 Bit build 
# Builds everything!
#
####################################################################################################################

SET_PROPERTY (GLOBAL PROPERTY SubProject Albany32Bit)
SET_PROPERTY (GLOBAL PROPERTY Label Albany32Bit)

SET (ALB_LAME_DIR "/projects/albany/src/lame-4.24.1/")

SET(CONFIGURE_OPTIONS
  "-DALBANY_TRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
  "-DENABLE_LCM:BOOL=ON"
  "-DENABLE_CONTACT:BOOL=ON"
  "-DENABLE_LCM_SPECULATIVE:BOOL=OFF"
  "-DENABLE_HYDRIDE:BOOL=ON"
  "-DENABLE_SCOREC:BOOL=ON"
  "-DENABLE_SG_MP:BOOL=OFF"
  "-DENABLE_FELIX:BOOL=ON"
  "-DENABLE_AERAS:BOOL=ON"
  "-DENABLE_QCAD:BOOL=ON"
  "-DENABLE_MOR:BOOL=ON"
  "-DENABLE_ATO:BOOL=ON"
  "-DENABLE_AMP:BOOL=ON"
  "-DENABLE_ASCR:BOOL=OFF"
  "-DENABLE_CHECK_FPE:BOOL=ON"
  "-DLAME_INCLUDE_DIR:FILEPATH=${ALB_LAME_DIR}/include"
  "-DLAME_LIBRARY_DIR:FILEPATH=${ALB_LAME_DIR}/build"
  "-DENABLE_LAME:BOOL=ON"
   )
 
if(NOT EXISTS "${CTEST_BINARY_DIRECTORY}/Albany32Bit")
  FILE(MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/Albany32Bit)
endif()

CTEST_CONFIGURE(
          BUILD "${CTEST_BINARY_DIRECTORY}/Albany32Bit"
          SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
          OPTIONS "${CONFIGURE_OPTIONS}"
          RETURN_VALUE HAD_ERROR
          APPEND
)

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(PARTS Configure
               RETURN_VALUE  S_HAD_ERROR
  )

  if(S_HAD_ERROR)
	message(FATAL_ERROR "Cannot submit Albany configure results!")
  endif()
ENDIF()

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot configure Albany build!")
endif()

#
# Build Albany
#
###################################################################################################################

SET(CTEST_BUILD_TARGET all)

MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

CTEST_BUILD(
          BUILD "${CTEST_BINARY_DIRECTORY}/Albany32Bit"
          RETURN_VALUE  HAD_ERROR
          NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
          APPEND
)

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(PARTS Build
               RETURN_VALUE  S_HAD_ERROR
  )

  if(S_HAD_ERROR)
	message(FATAL_ERROR "Cannot submit Albany build results!")
  endif()
ENDIF()

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot build Albany!")
endif()

if(BUILD_LIBS_NUM_ERRORS GREATER 0)
	message(FATAL_ERROR "Encountered build errors in Albany build. Exiting!")
endif()

#
# Run Albany tests
#
##################################################################################################################

CTEST_TEST(
              BUILD "${CTEST_BINARY_DIRECTORY}/Albany32Bit"
#              PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
#              INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
              #NUMBER_FAILED  TEST_NUM_FAILED
               RETURN_VALUE  HAD_ERROR
)

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(PARTS Test
               RETURN_VALUE  S_HAD_ERROR
  )

  if(S_HAD_ERROR)
	message(FATAL_ERROR "Cannot submit Albany test results!")
  endif()
ENDIF()

#if(HAD_ERROR)
#	message(FATAL_ERROR "Some Albany tests failed.")
#endif()

ENDIF()

#
# Configure the Albany build using GO = long
#
########################################################################################################################

IF (BUILD_ALB64)
SET_PROPERTY (GLOBAL PROPERTY SubProject Albany64Bit)
SET_PROPERTY (GLOBAL PROPERTY Label Albany64Bit)

SET(CONFIGURE_OPTIONS
  "-DALBANY_TRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
  "-DENABLE_64BIT_INT:BOOL=ON"
  "-DENABLE_ALBANY_EPETRA_EXE:BOOL=OFF"
  "-DENABLE_LCM:BOOL=ON"
  "-DENABLE_LCM_SPECULATIVE:BOOL=OFF"
  "-DENABLE_HYDRIDE:BOOL=ON"
  "-DENABLE_SCOREC:BOOL=ON"
  "-DENABLE_SG_MP:BOOL=OFF"
  "-DENABLE_QCAD:BOOL=OFF"
  "-DENABLE_MOR:BOOL=OFF"
  "-DENABLE_CHECK_FPE:BOOL=ON"
  )

if(NOT EXISTS "${CTEST_BINARY_DIRECTORY}/Albany64Bit")
  FILE(MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/Albany64Bit)
endif()

#
# The 64 bit build 
#
##########################################################################################################################

CTEST_CONFIGURE(
          BUILD "${CTEST_BINARY_DIRECTORY}/Albany64Bit"
          SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
          OPTIONS "${CONFIGURE_OPTIONS}"
          RETURN_VALUE HAD_ERROR
          APPEND
)

# Read the CTestCustom.cmake file to turn off ignored tests

#CTEST_READ_CUSTOM_FILES("${CTEST_BINARY_DIRECTORY}/AlbanyT64")

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(PARTS Configure
               RETURN_VALUE  S_HAD_ERROR
  )

  if(S_HAD_ERROR)
	message(FATAL_ERROR "Cannot submit Albany 64 bit configure results!")
  endif()
ENDIF()

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot configure Albany 64 bit build!")
endif()

#
# Build Albany 64 bit
#
#######################################################################################################################

SET(CTEST_BUILD_TARGET all)

MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

CTEST_BUILD(
          BUILD "${CTEST_BINARY_DIRECTORY}/Albany64Bit"
          RETURN_VALUE  HAD_ERROR
          NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
          APPEND
)

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(PARTS Build
               RETURN_VALUE  S_HAD_ERROR
  )

  if(S_HAD_ERROR)
	message(FATAL_ERROR "Cannot submit Albany 64 bit build results!")
  endif()
ENDIF()

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot build Albany 64 bit!")
endif()

if(BUILD_LIBS_NUM_ERRORS GREATER 0)
	message(FATAL_ERROR "Encountered build errors in Albany 64 bit build. Exiting!")
endif()
#
# Run Albany 64 bit tests
#
#############################################################################################################################

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
	message(FATAL_ERROR "Cannot submit Albany 64 bit test results!")
  endif()
ENDIF()
ENDIF()

# Add the path to Clang libraries needed for the Clang configure, build and sest cycle
#
# Need to add the openmpi libraries at the front of LD_LIBRARY_PATH
#

set(ENV{LD_LIBRARY_PATH} 
/projects/albany/clang/lib:${INITIAL_LD_LIBRARY_PATH}
)

IF(BUILD_TRILINOSCLANG11)

#
# Configure the Trilinos/SCOREC Clang build
#
####################################################################################################################

SET_PROPERTY (GLOBAL PROPERTY SubProject TrilinosClang++11)
SET_PROPERTY (GLOBAL PROPERTY Label TrilinosClang++11)

SET(CONFIGURE_OPTIONS
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
  ${COMMON_CONFIGURE_OPTIONS}
  )

if(NOT EXISTS "${CTEST_BINARY_DIRECTORY}/TriBuildC11")
  FILE(MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/TriBuildC11)
endif()

CTEST_CONFIGURE(
          BUILD "${CTEST_BINARY_DIRECTORY}/TriBuildC11"
          SOURCE "${CTEST_SOURCE_DIRECTORY}/Trilinos"
          OPTIONS "${CONFIGURE_OPTIONS}"
          RETURN_VALUE HAD_ERROR
)

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(PARTS Configure
               RETURN_VALUE  S_HAD_ERROR
  )

  if(S_HAD_ERROR)
	message(FATAL_ERROR "Cannot submit TrilinosClang++11 configure results!")
  endif()
ENDIF()

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot configure TrilinosClang++11 build!")
endif()

#SET(CTEST_BUILD_TARGET all)
SET(CTEST_BUILD_TARGET install)

MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

CTEST_BUILD(
          BUILD "${CTEST_BINARY_DIRECTORY}/TriBuildC11"
          RETURN_VALUE  HAD_ERROR
          NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
          APPEND
)

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(PARTS Build
               RETURN_VALUE  S_HAD_ERROR
  )

  if(S_HAD_ERROR)
	message(FATAL_ERROR "Cannot submit TrilinoClang++11 build results!")
  endif()

ENDIF()

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot build Trilinos with Clang!")
endif()

if(BUILD_LIBS_NUM_ERRORS GREATER 0)
	message(FATAL_ERROR "Encountered build errors in Trilinos Clang build. Exiting!")
endif()

ENDIF()

#
# Configure the Albany Clang build using GO = long
#
##############################################################################################################################

IF (BUILD_ALB64CLANG11)
SET_PROPERTY (GLOBAL PROPERTY SubProject Albany64BitClang++11)
SET_PROPERTY (GLOBAL PROPERTY Label Albany64BitClang++11)

SET(CONFIGURE_OPTIONS
  "-DALBANY_TRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstallC11"
  "-DENABLE_64BIT_INT:BOOL=ON"
  "-DENABLE_ALBANY_EPETRA_EXE:BOOL=OFF"
  "-DENABLE_LCM:BOOL=ON"
  "-DENABLE_LCM_SPECULATIVE:BOOL=OFF"
  "-DENABLE_HYDRIDE:BOOL=ON"
  "-DENABLE_SCOREC:BOOL=ON"
  "-DENABLE_SG_MP:BOOL=OFF"
  "-DENABLE_QCAD:BOOL=OFF"
  "-DENABLE_MOR:BOOL=OFF"
  "-DENABLE_CHECK_FPE:BOOL=ON"
  )

if(NOT EXISTS "${CTEST_BINARY_DIRECTORY}/Albany64BitC11")
  FILE(MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/Albany64BitC11)
endif()

#
# The Clang 64 bit build 
#
##########################################################################################################################

CTEST_CONFIGURE(
          BUILD "${CTEST_BINARY_DIRECTORY}/Albany64BitC11"
          SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
          OPTIONS "${CONFIGURE_OPTIONS}"
          RETURN_VALUE HAD_ERROR
          APPEND
)

# Read the CTestCustom.cmake file to turn off ignored tests

#CTEST_READ_CUSTOM_FILES("${CTEST_BINARY_DIRECTORY}/AlbanyT64")

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(PARTS Configure
               RETURN_VALUE  S_HAD_ERROR
  )

  if(S_HAD_ERROR)
	message(FATAL_ERROR "Cannot submit Albany 64 bit Clang configure results!")
  endif()
ENDIF()

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot configure Albany 64 bit Clang build!")
endif()

#
# Build Clang Albany 64 bit
#
#######################################################################################################################

SET(CTEST_BUILD_TARGET all)

MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

CTEST_BUILD(
          BUILD "${CTEST_BINARY_DIRECTORY}/Albany64BitC11"
          RETURN_VALUE  HAD_ERROR
          NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
          APPEND
)

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(PARTS Build
               RETURN_VALUE  S_HAD_ERROR
  )

  if(S_HAD_ERROR)
	message(FATAL_ERROR "Cannot submit Albany 64 bit Clang build results!")
  endif()
ENDIF()

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot build Albany 64 bit with Clang!")
endif()

if(BUILD_LIBS_NUM_ERRORS GREATER 0)
	message(FATAL_ERROR "Encountered build errors in Albany 64 bit Clang build. Exiting!")
endif()
#
# Run Clang Albany 64 bit tests
#
########################################################################################################################

CTEST_TEST(
              BUILD "${CTEST_BINARY_DIRECTORY}/Albany64BitC11"
#              PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
#              INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
              #NUMBER_FAILED  TEST_NUM_FAILED
)

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(PARTS Test
               RETURN_VALUE  HAD_ERROR
  )

  if(HAD_ERROR)
	message(FATAL_ERROR "Cannot submit Albany 64 bit Clang test results!")
  endif()
ENDIF()
ENDIF()

# Return the LD_LIBRARY_PATH back to the initial one 

#set(ENV{LD_LIBRARY_PATH} ${INITIAL_LD_LIBRARY_PATH})


# Done!!!

