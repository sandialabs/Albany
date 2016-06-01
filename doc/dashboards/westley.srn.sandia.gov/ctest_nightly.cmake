cmake_minimum_required(VERSION 2.8)

SET(CTEST_DO_SUBMIT "$ENV{DO_SUBMIT}")
SET(CTEST_TEST_TYPE "$ENV{TEST_TYPE}")

# What to build and test
SET(DOWNLOAD_TRILINOS TRUE)
SET(DOWNLOAD_ALBANY TRUE)
SET(BUILD_TRILINOS TRUE)
SET(BUILD_ALBANY TRUE)
SET(CLEAN_BUILD TRUE)

# Begin User inputs:
set( CTEST_SITE             "westley.srn.sandia.gov" ) # generally the output of hostname
set( CTEST_DASHBOARD_ROOT   "$ENV{TEST_DIRECTORY}" ) # writable path
set( CTEST_SCRIPT_DIRECTORY   "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set( CTEST_CMAKE_GENERATOR  "Unix Makefiles" ) # What is your compilation apps ?
set( CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?

set( CTEST_PROJECT_NAME         "Albany" )
set( CTEST_SOURCE_NAME          repos)
set( CTEST_BUILD_NAME           "intel-mic-${CTEST_BUILD_CONFIGURATION}")
set( CTEST_BINARY_NAME          buildAlbany)

# Double the test time because MICs are pretty slow
set( CTEST_TEST_TIMEOUT 1200)

SET(PREFIX_DIR /home/gahanse)
SET(CMAKE_SW_INSTALL_DIR /usr/local)
SET(MPI_BASE_DIR /opt/intel/impi/5.0.3.048/intel64)
SET(BOOST_DIR /usr/local/mic/boost-1.58.0)
SET(NETCDF /usr/local/mic)
SET(HDFDIR /usr/local/mic)
SET(ZLIB_DIR /usr/local/mic)
SET(PARMETISDIR /usr/local/mic)
SET(HWLOC_PATH /usr/local/mic)
SET(INTEL_DIR /opt/intel/mkl/lib/mic)

SET (CTEST_SOURCE_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_SOURCE_NAME}")
SET (CTEST_BINARY_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_BINARY_NAME}")

IF (CLEAN_BUILD)
  IF(EXISTS "${CTEST_BINARY_DIRECTORY}" )
    FILE(REMOVE_RECURSE "${CTEST_BINARY_DIRECTORY}")
  ENDIF()
ENDIF()

IF(NOT EXISTS "${CTEST_SOURCE_DIRECTORY}")
  FILE(MAKE_DIRECTORY "${CTEST_SOURCE_DIRECTORY}")
ENDIF()

IF(NOT EXISTS "${CTEST_BINARY_DIRECTORY}")
  FILE(MAKE_DIRECTORY "${CTEST_BINARY_DIRECTORY}")
ENDIF()

configure_file(${CTEST_SCRIPT_DIRECTORY}/CTestConfig.cmake
               ${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake COPYONLY)

# Run test at/after 21:00 (9:00PM MDT --> 3:00 UTC, 8:00PM MST --> 3:00 UTC)
SET (CTEST_NIGHTLY_START_TIME "03:00:00 UTC")
SET (CTEST_CMAKE_COMMAND "${CMAKE_SW_INSTALL_DIR}/bin/cmake")
SET (CTEST_COMMAND "${CMAKE_SW_INSTALL_DIR}/bin/ctest -D ${CTEST_TEST_TYPE}")
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

# Point at the public Repo
SET(Trilinos_REPOSITORY_LOCATION https://github.com/trilinos/Trilinos.git)
SET(SCOREC_REPOSITORY_LOCATION https://github.com/SCOREC/core.git)
SET(Albany_REPOSITORY_LOCATION git@github.com:gahansen/Albany.git)

SET(TRILINOS_HOME "${CTEST_SOURCE_DIRECTORY}/Trilinos")

IF (DOWNLOAD_TRILINOS)
#
# Get the Trilinos repo
#
#########################################################################################################

set(CTEST_CHECKOUT_COMMAND)

if(NOT EXISTS "${TRILINOS_HOME}")
  EXECUTE_PROCESS(COMMAND "${CTEST_GIT_COMMAND}" 
    clone ${Trilinos_REPOSITORY_LOCATION} ${TRILINOS_HOME}
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

if(NOT EXISTS "${TRILINOS_HOME}/SCOREC")
  EXECUTE_PROCESS(COMMAND "${CTEST_GIT_COMMAND}"
    clone ${SCOREC_REPOSITORY_LOCATION} ${TRILINOS_HOME}/SCOREC
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

ENDIF()

IF (DOWNLOAD_ALBANY)

#
# Get ALBANY
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

CTEST_START(${CTEST_TEST_TYPE})

#
# Send the project structure to CDash
#
##############################################################################################################

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(FILES "${CTEST_SCRIPT_DIRECTORY}/Project.xml"
               RETURN_VALUE  HAD_ERROR
  )

  if(HAD_ERROR)
    message(FATAL_ERROR "Cannot submit miniContact Project.xml!")
  endif()
ENDIF()

IF(DOWNLOAD_TRILINOS)

#
# Update Trilinos
#
###########################################################################################################

SET_PROPERTY (GLOBAL PROPERTY SubProject Trilinos_MIC)
SET_PROPERTY (GLOBAL PROPERTY Label Trilinos_MIC)

set(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
CTEST_UPDATE(SOURCE "${TRILINOS_HOME}" RETURN_VALUE count)
message("Found ${count} changed files")

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(PARTS Update
               RETURN_VALUE  HAD_ERROR
  )

  if(HAD_ERROR)
    message(FATAL_ERROR "Cannot submit Trilinos update results!")
  endif()
ENDIF()

IF(count LESS 0)
        message(FATAL_ERROR "Cannot update Trilinos!")
endif()

# Get the SCOREC tools

set(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
CTEST_UPDATE(SOURCE "${TRILINOS_HOME}/SCOREC" RETURN_VALUE count)
message("Found ${count} changed files")

IF(count LESS 0)
        message(FATAL_ERROR "Cannot update SCOREC tools!")
endif()

ENDIF()

IF(DOWNLOAD_ALBANY)

#
# Update Albany
#
##############################################################################################################

SET_PROPERTY (GLOBAL PROPERTY SubProject Albany_MIC)
SET_PROPERTY (GLOBAL PROPERTY Label Albany_MIC)

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
        message(FATAL_ERROR "Cannot update Albany!")
endif()

ENDIF()

#
# Set the common Trilinos config options
#
#######################################################################################################################

SET(CONFIGURE_OPTIONS
  -Wno-dev
  -DTrilinos_CONFIGURE_OPTIONS_FILE:FILEPATH=${TRILINOS_HOME}/sampleScripts/AlbanySettings.cmake
  -DTrilinos_ENABLE_SCOREC:BOOL=ON
  -DSCOREC_DISABLE_STRONG_WARNINGS:BOOL=ON
  -DCMAKE_BUILD_TYPE:STRING=NONE
  -DCMAKE_CXX_COMPILER:FILEPATH=${MPI_BASE_DIR}/bin/mpiicpc
  -DCMAKE_C_COMPILER:FILEPATH=${MPI_BASE_DIR}/bin/mpiicc
  -DCMAKE_Fortran_COMPILER:FILEPATH=${MPI_BASE_DIR}/bin/mpiifort
  -DCMAKE_AR:FILEPATH=/opt/intel/composer_xe_2015.3.187/bin/intel64_mic/xiar
  -DCMAKE_LINKER:FILEPATH=/opt/intel/composer_xe_2015.3.187/bin/intel64_mic/xild
  -DTrilinos_SHOW_DEPRECATED_WARNINGS:BOOL=OFF
  "-DCMAKE_CXX_FLAGS:STRING='-O3 -w -mmic -mkl=sequential -mt_mpi -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX -DPREC_TIMER -restrict -fasm-blocks -DDEVICE=1wq  -fopenmp -DBUILD_PHALANX_FOR_ALBANY'"
  "-DCMAKE_C_FLAGS:STRING='-O3 -w -mmic -mkl=sequential -mt_mpi -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX -DPREC_TIMER -restrict -fasm-blocks -DDEVICE=1wq  -fopenmp'"
  "-DCMAKE_Fortran_FLAGS:STRING='-O3 -w -mmic -mkl=sequential -mt_mpi -DPREC_TIMER -fopenmp'"
  -DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON
  -DTpetra_INST_INT_LONG_LONG:BOOL=OFF
  -DTpetra_INST_INT_INT:BOOL=ON
  -DTpetra_INST_DOUBLE:BOOL=ON
  -DTpetra_INST_FLOAT:BOOL=OFF
  -DTpetra_INST_SERIAL:BOOL=ON
  -DTpetra_INST_COMPLEX_FLOAT:BOOL=OFF
  -DTpetra_INST_COMPLEX_DOUBLE:BOOL=OFF
  -DTpetra_INST_INT_LONG:BOOL=OFF
  -DTpetra_INST_INT_UNSIGNED:BOOL=OFF
  -DZoltan_ENABLE_ULONG_IDS:BOOL=OFF
  -DTeuchos_ENABLE_LONG_LONG_INT:BOOL=ON
#
  -DTrilinos_ENABLE_Kokkos:BOOL=ON
  -DTrilinos_ENABLE_KokkosCore:BOOL=ON
  -DPhalanx_KOKKOS_DEVICE_TYPE:STRING=OPENMP
  -DPhalanx_INDEX_SIZE_TYPE:STRING=INT
  -DKokkos_ENABLE_Serial:BOOL=ON
  -DKokkos_ENABLE_OpenMP:BOOL=ON
  -DKokkos_ENABLE_Pthread:BOOL=OFF
  -DKokkos_ENABLE_Cuda:BOOL=OFF
  -DTrilinos_ENABLE_OpenMP:BOOL=ON
  -DTPL_ENABLE_CUDA:BOOL=OFF
  -DTPL_ENABLE_CUSPARSE:BOOL=OFF
#
  -DTPL_ENABLE_MPI:BOOL=ON
  -DMPI_BASE_DIR:PATH=${MPI_BASE_DIR}
#
  -DTPL_ENABLE_Pthread:BOOL=OFF
  -DTPL_ENABLE_HWLOC:BOOL=OFF
#
  -DTPL_ENABLE_Boost:BOOL=ON
  -DTPL_ENABLE_BoostLib:BOOL=ON
  -DTPL_ENABLE_BoostAlbLib:BOOL=ON
  -DBoost_INCLUDE_DIRS:PATH=${BOOST_DIR}/include
  -DBoost_LIBRARY_DIRS:PATH=${BOOST_DIR}/lib
  -DBoostLib_INCLUDE_DIRS:PATH=${BOOST_DIR}/include
  -DBoostLib_LIBRARY_DIRS:PATH=${BOOST_DIR}/lib
  -DBoostAlbLib_INCLUDE_DIRS:PATH=${BOOST_DIR}/include
  -DBoostAlbLib_LIBRARY_DIRS:PATH=${BOOST_DIR}/lib
#
  -DTPL_ENABLE_Netcdf:STRING=ON
  -DTPL_Netcdf_PARALLEL:BOOL=ON
  -DNetcdf_INCLUDE_DIRS:PATH=${NETCDF}/include
  "-DTPL_Netcdf_LIBRARIES:FILEPATH='${NETCDF}/lib64/libnetcdf.a\\;${HDFDIR}/lib/libhdf5_hl.a\\;${HDFDIR}/lib/libhdf5.a\\;${ZLIB_DIR}/lib/libz.a'"
#
  -DTPL_ENABLE_HDF5:STRING=ON
  -DHDF5_INCLUDE_DIRS:PATH=${HDFDIR}/include
  "-DTPL_HDF5_LIBRARIES:FILEPATH='${NETCDF}/lib64/libnetcdf.a\\;${HDFDIR}/lib/libhdf5_hl.a\\;${HDFDIR}/lib/libhdf5.a\\;${ZLIB_DIR}/lib/libz.a'"
#
  -DTPL_ENABLE_Zlib:STRING=ON
  -DZlib_INCLUDE_DIRS:PATH=${ZLIB_DIR}/include
  -DZlib_LIBRARY_DIRS:PATH=${ZLIB_DIR}/lib
#
  -DTPL_ENABLE_BLAS:BOOL=ON
  -DTPL_ENABLE_LAPACK:BOOL=ON
  -DBLAS_LIBRARY_DIRS:FILEPATH=${INTEL_DIR}
  "-DTPL_BLAS_LIBRARIES:STRING='${INTEL_DIR}/libmkl_intel_lp64.a\\;${INTEL_DIR}/libmkl_sequential.a\\;${INTEL_DIR}/libmkl_core.a'"
  -DLAPACK_LIBRARY_NAMES:STRING=
#
  -DTPL_ENABLE_ParMETIS:STRING=ON
  -DParMETIS_INCLUDE_DIRS:PATH=${PARMETISDIR}/include
  -DParMETIS_LIBRARY_DIRS:PATH=${PARMETISDIR}/lib
#
  -DCMAKE_INSTALL_PREFIX:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall
#
  -DTrilinos_ENABLE_Moertel:BOOL=OFF
  -DTrilinos_ENABLE_TriKota:BOOL=OFF
  -DSEACAS_ENABLE_SEACASSVDI:BOOL=OFF
  -DTrilinos_ENABLE_SEACASPLT:BOOL=OFF
  -DTrilinos_ENABLE_SEACASBlot:BOOL=OFF
  -DTPL_ENABLE_X11:BOOL=OFF
  -DTPL_ENABLE_Matio:BOOL=OFF
  -DTrilinos_ENABLE_ThreadPool:BOOL=OFF
  -DTrilinos_ENABLE_Teko:BOOL=OFF
  -DTrilinos_ENABLE_MueLu:BOOL=ON
# Comment these out to disable stk
  -DTrilinos_ENABLE_STK:BOOL=ON
  -DTrilinos_ENABLE_STKUtil:BOOL=ON
  -DTrilinos_ENABLE_STKTopology:BOOL=ON
  -DTrilinos_ENABLE_STKMesh:BOOL=ON
  -DTrilinos_ENABLE_STKIO:BOOL=ON
  -DTrilinos_ENABLE_STKTransfer:BOOL=ON
# Comment these out to enable stk
#  -DTrilinos_ENABLE_STK:BOOL=OFF
#  -DTrilinos_ENABLE_SEACAS:BOOL=OFF
#
  -DTrilinos_ENABLE_Amesos2:BOOL=ON
  -DAmesos2_ENABLE_KLU2:BOOL=ON
# Try turning off more of Trilinos
  -DTrilinos_ENABLE_OptiPack:BOOL=OFF
  -DTrilinos_ENABLE_GlobiPack:BOOL=OFF
#  -DTPL_FIND_SHARED_LIBS:BOOL=OFF
  -DBUILD_SHARED_LIBS:BOOL=OFF
  )

IF(BUILD_TRILINOS)

#
# Configure the Trilinos build
#
###############################################################################################################

SET_PROPERTY (GLOBAL PROPERTY SubProject Trilinos_MIC)
SET_PROPERTY (GLOBAL PROPERTY Label Trilinos_MIC)

if(NOT EXISTS "${CTEST_BINARY_DIRECTORY}/TriBuild")
  FILE(MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/TriBuild)
endif()

IF (CLEAN_BUILD)
# Initial cache info
set( CACHE_CONTENTS "
SITE:STRING=${CTEST_SITE}
CMAKE_BUILD_TYPE:STRING=Release
CMAKE_GENERATOR:INTERNAL=${CTEST_CMAKE_GENERATOR}
BUILD_TESTING:BOOL=OFF
PRODUCT_REPO:STRING=${Trilinos_REPOSITORY_LOCATION}
" )
file(WRITE "${CTEST_BINARY_DIRECTORY}/TriBuild/CMakeCache.txt" "${CACHE_CONTENTS}")
ENDIF(CLEAN_BUILD)


CTEST_CONFIGURE(
          BUILD "${CTEST_BINARY_DIRECTORY}/TriBuild"
          SOURCE "${TRILINOS_HOME}"
          OPTIONS "${CONFIGURE_OPTIONS}"
          RETURN_VALUE HAD_ERROR
          APPEND
)

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(PARTS Configure
               RETURN_VALUE  S_HAD_ERROR
  )

  if(S_HAD_ERROR)
    message(FATAL_ERROR "Cannot submit Trilinos configure results!")
  endif()
ENDIF(CTEST_DO_SUBMIT)

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot configure Trilinos build!")
endif()

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
    message(FATAL_ERROR "Cannot submit Trilinos build results!")
  endif()

ENDIF(CTEST_DO_SUBMIT)

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot build Trilinos!")
endif()

if(BUILD_LIBS_NUM_ERRORS GREATER 0)
        message(FATAL_ERROR "Encountered build errors in Trilinos build. Exiting!")
endif()

ENDIF(BUILD_TRILINOS)

IF (BUILD_ALBANY)

# Configure the ALBANY build 
#
####################################################################################################################

SET_PROPERTY (GLOBAL PROPERTY SubProject Albany_MIC)
SET_PROPERTY (GLOBAL PROPERTY Label Albany_MIC)

SET(CONFIGURE_OPTIONS
  "-DALBANY_TRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
  "-DENABLE_LCM:BOOL=ON"
  "-DENABLE_LCM_SPECULATIVE:BOOL=OFF"
  "-DENABLE_HYDRIDE:BOOL=OFF"
  "-DENABLE_SCOREC:BOOL=ON"
  "-DENABLE_SG:BOOL=OFF"
  "-DENABLE_ENSEMBLE:BOOL=OFF"
  "-DENABLE_FELIX:BOOL=ON"
  "-DENABLE_AERAS:BOOL=ON"
  "-DENABLE_QCAD:BOOL=OFF"
  "-DENABLE_MOR:BOOL=OFF"
  "-DENABLE_ATO:BOOL=OFF"
  "-DENABLE_ASCR:BOOL=OFF"
  "-DENABLE_CHECK_FPE:BOOL=OFF"
  "-DENABLE_LAME:BOOL=OFF"
  "-DENABLE_BGL:BOOL=OFF"
  "-DENABLE_ALBANY_EPETRA_EXE:BOOL=ON"
  "-DENABLE_KOKKOS_UNDER_DEVELOPMENT:BOOL=ON"
  "-DENABLE_CROSS_COMPILE:BOOL=ON"
  "-DALBANY_MPI_OPTIONS:BOOL=ON"
  "-DALBANY_MPI_EXEC:STRING=${MPI_BASE_DIR}/bin/mpiexec.hydra"
  "-DALBANY_MPI_EXEC_NUMPROCS_FLAG:STRING=-n"
  "-DALBANY_MPI_EXEC_MAX_NUMPROCS:STRING=4"
  "-DALBANY_MPI_TRAILING_OPTIONS:STRING='-hosts mic1 -ppn 4 -env OMP_NUM_THREADS 56 -env KMP_AFFINITY balanced -binding domain=omp -env LD_LIBRARY_PATH /opt/intel/mkl/lib/mic'"
   )
 
if(NOT EXISTS "${CTEST_BINARY_DIRECTORY}/Albany")
  FILE(MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/Albany)
endif()

IF (CLEAN_BUILD)
# Initial cache info
set( CACHE_CONTENTS "
SITE:STRING=${CTEST_SITE}
CMAKE_BUILD_TYPE:STRING=Release
CMAKE_GENERATOR:INTERNAL=${CTEST_CMAKE_GENERATOR}
BUILD_TESTING:BOOL=OFF
PRODUCT_REPO:STRING=${Albany_REPOSITORY_LOCATION}
" )
file(WRITE "${CTEST_BINARY_DIRECTORY}/Albany/CMakeCache.txt" "${CACHE_CONTENTS}")
ENDIF(CLEAN_BUILD)


CTEST_CONFIGURE(
          BUILD "${CTEST_BINARY_DIRECTORY}/Albany"
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
          BUILD "${CTEST_BINARY_DIRECTORY}/Albany"
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
ENDIF(CTEST_DO_SUBMIT)

if(HAD_ERROR)
  message(FATAL_ERROR "Cannot build Albany!")
endif()

#
# Run Albany tests
#
##################################################################################################################

CTEST_TEST(
              BUILD "${CTEST_BINARY_DIRECTORY}/Albany"
#              INCLUDE "SCOREC_ThermoMechanicalCan_thermomech_tpetra"
#              PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
#              INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
#              INCLUDE_LABEL "CUDA_TEST"
              #NUMBER_FAILED  TEST_NUM_FAILED
)

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(PARTS Test
               RETURN_VALUE  HAD_ERROR
  )

  if(HAD_ERROR)
    message(FATAL_ERROR "Cannot submit Albany test results!")
  endif()
ENDIF(CTEST_DO_SUBMIT)

ENDIF (BUILD_ALBANY)

# Done!!!
