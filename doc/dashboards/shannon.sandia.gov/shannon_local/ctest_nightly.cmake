cmake_minimum_required(VERSION 2.8)

#SET(CTEST_DO_SUBMIT ON)
#SET(CTEST_TEST_TYPE Nightly)

#SET(CTEST_DO_SUBMIT OFF)
#SET(CTEST_TEST_TYPE Experimental)

SET(CTEST_DO_SUBMIT "$ENV{DO_SUBMIT}")
SET(CTEST_TEST_TYPE "$ENV{TEST_TYPE}")

# What to build and test
SET(DOWNLOAD_TRILINOS TRUE)
SET(DOWNLOAD_ALBANY TRUE)
SET(BUILD_TRILINOS TRUE)
SET(BUILD_ALBANY TRUE)
SET(CLEAN_BUILD TRUE)

# Begin User inputs:
set( CTEST_SITE             "shannon.sandia.gov" ) # generally the output of hostname
set( CTEST_DASHBOARD_ROOT   "$ENV{TEST_DIRECTORY}" ) # writable path
set( CTEST_SCRIPT_DIRECTORY   "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set( CTEST_CMAKE_GENERATOR  "Unix Makefiles" ) # What is your compilation apps ?
set( CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?

set( CTEST_PROJECT_NAME         "Albany" )
set( CTEST_SOURCE_NAME          repos)
set( CTEST_BUILD_NAME           "cuda-nvcc-${CTEST_BUILD_CONFIGURATION}")
set( CTEST_BINARY_NAME          buildAlbany)

SET(PREFIX_DIR /home/gahanse)
SET(MPI_BASE_DIR /home/projects/x86-64/openmpi/1.8.4/gnu/4.7.2/cuda/7.0.28)
SET(INTEL_DIR /opt/intel/mkl/lib/intel64)
#SET(BOOST_DIR /home/projects/x86-64/boost/1.57.0/gnu/4.7.2)
SET(BOOST_DIR /home/gahanse)


SET (CTEST_SOURCE_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_SOURCE_NAME}")
SET (CTEST_BINARY_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_BINARY_NAME}")

IF (CLEAN_BUILD)
  IF(EXISTS "${CTEST_BINARY_DIRECTORY}" )
#  ctest_empty_binary_directory( "${CTEST_BINARY_DIRECTORY}" )
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

# Rougly midnight 0700 UTC
SET(CTEST_NIGHTLY_START_TIME "07:00:00 UTC")
SET (CTEST_CMAKE_COMMAND "${PREFIX_DIR}/bin/cmake")
SET (CTEST_COMMAND "${PREFIX_DIR}/bin/ctest -D ${CTEST_TEST_TYPE}")
SET (CTEST_BUILD_FLAGS "-j16")

set(CTEST_DROP_SITE "shannon.sandia.gov")
set(CTEST_DROP_LOCATION "nightly/Albany")
set(CTEST_DROP_METHOD "cp")
set(CTEST_TRIGGER_SITE "")
set(CTEST_DROP_SITE_USER "")

find_program(CTEST_GIT_COMMAND NAMES git)

# Point at the public Repo
SET(Trilinos_REPOSITORY_LOCATION software.sandia.gov:/git/Trilinos)
SET(SCOREC_REPOSITORY_LOCATION https://github.com/SCOREC/core.git)
SET(Albany_REPOSITORY_LOCATION https://github.com/gahansen/Albany.git)

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

IF(DOWNLOAD_TRILINOS)

#
# Update Trilinos
#
###########################################################################################################

SET_PROPERTY (GLOBAL PROPERTY SubProject Trilinos_CUVM)
SET_PROPERTY (GLOBAL PROPERTY Label Trilinos_CUVM)

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

# Save a copy of the Trilinos update to post to the CDash site.

#EXECUTE_PROCESS( COMMAND ${CTEST_SCP_COMMAND} ${CTEST_DROP_SITE}:${CTEST_DROP_LOCATION}/Update.xml ${CTEST_DROP_SITE}:${CTEST_DROP_LOCATION}/Update_Trilinos.xml
#               )

# Note: CTest will store files in ${CTEST_BINARY_DIRECTORY}/${CTEST_DROP_LOCATION} - ending in Update.xml.
# Not sure what the first part of that filename is, so glob all possibliities into UPDATE_FILES. Then
# grab the last one and rename it to Update_Trilinos.xml in the drop location. We assume there is only a 
# single update file, if there are more skip this.
FILE(GLOB UPDATE_FILES "${CTEST_BINARY_DIRECTORY}/${CTEST_DROP_LOCATION}/*Update.xml")
LIST(LENGTH UPDATE_FILES UP_LIST_LEN)
IF(UP_LIST_LEN EQUAL 1)
  LIST(GET UPDATE_FILES -1 SINGLE_UPDATE_FILE)
  FILE(RENAME "${SINGLE_UPDATE_FILE}" "${CTEST_BINARY_DIRECTORY}/${CTEST_DROP_LOCATION}/Update_Trilinos.xml")
ENDIF()

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

SET_PROPERTY (GLOBAL PROPERTY SubProject Albany_CUVM)
SET_PROPERTY (GLOBAL PROPERTY Label Albany_CUVM)

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
  "-Wno-dev"
  "-DTrilinos_CONFIGURE_OPTIONS_FILE:FILEPATH=${TRILINOS_HOME}/sampleScripts/AlbanySettings.cmake"
  "-DTrilinos_EXTRA_REPOSITORIES:STRING=SCOREC"
  "-DTrilinos_ENABLE_SCOREC:BOOL=ON"
  "-DSCOREC_DISABLE_STRONG_WARNINGS:BOOL=ON"
  "-DCMAKE_BUILD_TYPE:STRING=NONE"
  "-DCMAKE_CXX_COMPILER:FILEPATH=${CTEST_SCRIPT_DIRECTORY}/nvcc_wrapper"
  "-DCMAKE_C_COMPILER:FILEPATH=mpicc"
  "-DCMAKE_Fortran_COMPILER:FILEPATH=mpifort"
  "-DCMAKE_CXX_FLAGS:STRING='-DNDEBUG'"
  "-DCMAKE_C_FLAGS:STRING='-O3 -w -DNDEBUG'"
  "-DCMAKE_Fortran_FLAGS:STRING='-O3 -w -DNDEBUG'"
  "-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON"
#  "-DTpetra_INST_INT_LONG_LONG:BOOL=ON"
  "-DTpetra_INST_INT_LONG_LONG:BOOL=OFF"
  "-DTpetra_INST_INT_INT:BOOL=ON"
  "-DTpetra_INST_DOUBLE:BOOL=ON"
  "-DTpetra_INST_FLOAT:BOOL=OFF"
  "-DTpetra_INST_COMPLEX_FLOAT:BOOL=OFF"
  "-DTpetra_INST_COMPLEX_DOUBLE:BOOL=OFF"
  "-DTpetra_INST_INT_LONG:BOOL=OFF"
  "-DTpetra_INST_INT_UNSIGNED:BOOL=OFF"
#
  "-DTrilinos_ENABLE_Kokkos:BOOL=ON"
  "-DPhalanx_KOKKOS_DEVICE_TYPE:STRING=CUDA"
  "-DPhalanx_INDEX_SIZE_TYPE:STRING=INT"
  "-DPhalanx_SHOW_DEPRECATED_WARNINGS:BOOL=OFF"
  "-DKokkos_ENABLE_Serial:BOOL=ON"
  "-DKokkos_ENABLE_OpenMP:BOOL=OFF"
  "-DKokkos_ENABLE_Pthread:BOOL=OFF"
  "-DKokkos_ENABLE_Cuda:BOOL=ON"
  "-DTPL_ENABLE_CUDA:BOOL=ON"
  "-DTPL_ENABLE_CUSPARSE:BOOL=ON"
  "-DKokkos_ENABLE_Cuda_UVM:BOOL=ON"
#
  "-DTPL_ENABLE_MPI:BOOL=ON"
  "-DMPI_BASE_DIR:PATH=${MPI_BASE_DIR}"
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
  "-DTPL_ENABLE_BLAS:BOOL=ON"
  "-DTPL_ENABLE_LAPACK:BOOL=ON"
  "-DBLAS_LIBRARY_DIRS:FILEPATH=${INTEL_DIR}"
  "-DTPL_BLAS_LIBRARIES:STRING='-L${INTEL_DIR} -lmkl_intel_lp64 -lmkl_sequential -lmkl_core'"
  "-DLAPACK_LIBRARY_NAMES:STRING=''"
#
  "-DTPL_ENABLE_ParMETIS:STRING=ON"
  "-DParMETIS_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DParMETIS_LIBRARY_DIRS:PATH=${PREFIX_DIR}/lib"
#
  "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF"
  "-DTrilinos_VERBOSE_CONFIGURE:BOOL=OFF"
#
  "-DTrilinos_EXTRA_LINK_FLAGS='-L${PREFIX_DIR}/lib -lnetcdf -lhdf5_hl -lhdf5 -lz'"
  "-DCMAKE_INSTALL_PREFIX:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
#
  "-DTrilinos_ENABLE_Moertel:BOOL=OFF"
  "-DTrilinos_ENABLE_TriKota:BOOL=OFF"
  "-DTPL_ENABLE_X11:BOOL=OFF"
  "-DTPL_ENABLE_Matio:BOOL=OFF"
  "-DTrilinos_ENABLE_ThreadPool:BOOL=OFF"
  "-DZoltan_ENABLE_ULONG_IDS:BOOL=OFF"
  "-DTeuchos_ENABLE_LONG_LONG_INT:BOOL=OFF"
  "-DTrilinos_ENABLE_Teko:BOOL=OFF"
  "-DTrilinos_ENABLE_MueLu:BOOL=ON"
# Comment these out to disable stk
  "-DTrilinos_ENABLE_SEACAS:BOOL=ON"
  "-DTrilinos_ENABLE_SEACASIoss:BOOL=ON"
  "-DTrilinos_ENABLE_SEACASExodus:BOOL=ON"
  "-DSEACAS_ENABLE_SEACASSVDI:BOOL=OFF"
  "-DTrilinos_ENABLE_SEACASFastq:BOOL=OFF"
  "-DTrilinos_ENABLE_SEACASBlot:BOOL=OFF"
  "-DTrilinos_ENABLE_SEACASPLT:BOOL=OFF"
  "-DTrilinos_ENABLE_STK:BOOL=ON"
  "-DTrilinos_ENABLE_STKClassic:BOOL=OFF"
  "-DTrilinos_ENABLE_STKTopology:BOOL=ON"
  "-DTrilinos_ENABLE_STKMesh:BOOL=ON"
  "-DTrilinos_ENABLE_STKIO:BOOL=ON"
  "-DTrilinos_ENABLE_STKTransfer:BOOL=ON"
# Comment these out to enable stk
#  "-DTrilinos_ENABLE_STK:BOOL=OFF"
#  "-DTrilinos_ENABLE_SEACAS:BOOL=OFF"
#
  "-DTrilinos_ENABLE_Amesos2:BOOL=ON"
  "-DAmesos2_ENABLE_KLU2:BOOL=ON"
# Try turning off more of Trilinos
  "-DTrilinos_ENABLE_OptiPack:BOOL=OFF"
  "-DTrilinos_ENABLE_GlobiPack:BOOL=OFF"
  "-DTrilinos_ENABLE_Pamgen:BOOL=OFF"
  )

IF(BUILD_TRILINOS)

#
# Configure the Trilinos build
#
###############################################################################################################

SET_PROPERTY (GLOBAL PROPERTY SubProject Trilinos_CUVM)
SET_PROPERTY (GLOBAL PROPERTY Label Trilinos_CUVM)

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
ENDIF()


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
ENDIF()

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot configure Trilinos build!")
endif()

# Save a copy of the Trilinos configure to post to the CDash site.

#EXECUTE_PROCESS( COMMAND ${CTEST_SCP_COMMAND} ${CTEST_DROP_SITE}:${CTEST_DROP_LOCATION}/Configure.xml ${CTEST_DROP_SITE}:${CTEST_DROP_LOCATION}/Configure_Trilinos.xml
#               )

FILE(GLOB CONFIG_FILES "${CTEST_BINARY_DIRECTORY}/${CTEST_DROP_LOCATION}/*Configure.xml")
LIST(LENGTH CONFIG_FILES CO_LIST_LEN)
IF(CO_LIST_LEN EQUAL 1)
  LIST(GET CONFIG_FILES -1 SINGLE_CONFIG_FILE)
  FILE(RENAME "${SINGLE_CONFIG_FILE}" "${CTEST_BINARY_DIRECTORY}/${CTEST_DROP_LOCATION}/Configure_Trilinos.xml")
ENDIF()

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

ENDIF()

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot build Trilinos!")
endif()

# Save a copy of the Trilinos build to post to the CDash site.

#EXECUTE_PROCESS( COMMAND ${CTEST_SCP_COMMAND} ${CTEST_DROP_SITE}:${CTEST_DROP_LOCATION}/Build.xml ${CTEST_DROP_SITE}:${CTEST_DROP_LOCATION}/Build_Trilinos.xml
#               )

FILE(GLOB BUILD_FILES "${CTEST_BINARY_DIRECTORY}/${CTEST_DROP_LOCATION}/*Build.xml")
LIST(LENGTH BUILD_FILES BU_LIST_LEN)
IF(BU_LIST_LEN EQUAL 1)
  LIST(GET BUILD_FILES -1 SINGLE_BUILD_FILE)
  FILE(RENAME "${SINGLE_BUILD_FILE}" "${CTEST_BINARY_DIRECTORY}/${CTEST_DROP_LOCATION}/Build_Trilinos.xml")
ENDIF()

if(BUILD_LIBS_NUM_ERRORS GREATER 0)
        message(FATAL_ERROR "Encountered build errors in Trilinos build. Exiting!")
endif()

ENDIF()

IF (BUILD_ALBANY)

# Configure the ALBANY build 
#
####################################################################################################################

SET_PROPERTY (GLOBAL PROPERTY SubProject Albany_CUVM)
SET_PROPERTY (GLOBAL PROPERTY Label Albany_CUVM)

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
  "-DENABLE_ALBANY_EPETRA_EXE:BOOL=ON"
  "-DENABLE_KOKKOS_UNDER_DEVELOPMENT:BOOL=ON"
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
ENDIF()


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

SET(CTEST_BUILD_TARGET "Albany")

MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

CTEST_BUILD(
          BUILD "${CTEST_BINARY_DIRECTORY}/Albany"
          RETURN_VALUE  HAD_ERROR
          NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
          APPEND
)

if(BUILD_LIBS_NUM_ERRORS GREATER 0)
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

  message(FATAL_ERROR "Encountered build errors in Albany build. Exiting!")

endif()

SET(CTEST_BUILD_TARGET "AlbanyT")

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
ENDIF()

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot build AlbanyT!")
endif()

if(BUILD_LIBS_NUM_ERRORS GREATER 0)
    message(FATAL_ERROR "Encountered build errors in AlbanyT build. Exiting!")
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
              INCLUDE_LABEL "CUDA_TEST"
              #NUMBER_FAILED  TEST_NUM_FAILED
)

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(PARTS Test
               RETURN_VALUE  HAD_ERROR
  )

  if(HAD_ERROR)
    message(FATAL_ERROR "Cannot submit Albany test results!")
  endif()
ENDIF()

ENDIF (BUILD_ALBANY)


# Done!!!

