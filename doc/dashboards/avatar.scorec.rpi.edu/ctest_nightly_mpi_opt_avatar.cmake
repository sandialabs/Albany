cmake_minimum_required(VERSION 2.8)

SET(CTEST_DO_SUBMIT ON)
SET(CTEST_TEST_TYPE Nightly)

#SET(CTEST_DO_SUBMIT OFF)
#SET(CTEST_TEST_TYPE Experimental)

# What to build and test
SET(BUILD_TRI_SCOREC TRUE)
SET(BUILD_TRI_NEW_STK TRUE)
SET(BUILD_ALB_BASE TRUE)
SET(BUILD_ALB_TPETRA TRUE)
SET(BUILD_ALB_TPETRA64 TRUE)

# Begin User inputs:
set( CTEST_SITE             "avatar.scorec.rpi.edu" ) # generally the output of hostname
set( CTEST_DASHBOARD_ROOT   "$ENV{TEST_DIRECTORY}" ) # writable path
set( CTEST_SCRIPT_DIRECTORY   "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set( CTEST_CMAKE_GENERATOR  "Unix Makefiles" ) # What is your compilation apps ?
set( CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?

set( CTEST_PROJECT_NAME         "Albany" )
set( CTEST_SOURCE_NAME          repos)
set( CTEST_BUILD_NAME           "linux-gcc-${CTEST_BUILD_CONFIGURATION}")
set( CTEST_BINARY_NAME          build)

SET(PREFIX_DIR /users/ghansen)

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
  SET(CTEST_DROP_SITE "my.cdash.com")
  SET(CTEST_PROJECT_NAME "Albany")
  SET(CTEST_DROP_LOCATION "/submit.php?project=Albany")
  SET(CTEST_TRIGGER_SITE "")
  SET(CTEST_DROP_SITE_CDASH TRUE)
ENDIF()

find_program(CTEST_GIT_COMMAND NAMES git)
find_program(CTEST_SVN_COMMAND NAMES svn)

# Point at the public Repo
SET(Trilinos_REPOSITORY_LOCATION https://software.sandia.gov/trilinos/repositories/publicTrilinos)
#SET(SCOREC_REPOSITORY_LOCATION https://redmine.scorec.rpi.edu/svn/buildutil/trunk/cmake)
#SET(Albany_REPOSITORY_LOCATION ghansen@jumpgate.scorec.rpi.edu:/users/ghansen/Albany.git)
SET(SCOREC_REPOSITORY_LOCATION git@github.com:SCOREC/core.git)
SET(Albany_REPOSITORY_LOCATION git@github.com:gahansen/Albany.git)

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


# Get the publicTrilinos repo

set(CTEST_CHECKOUT_COMMAND)

if(NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/publicTrilinos")
#  set(CTEST_CHECKOUT_COMMAND "${CTEST_GIT_COMMAND} clone ${Trilinos_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/publicTrilinos")
  EXECUTE_PROCESS(COMMAND "${CTEST_GIT_COMMAND}" 
    clone ${Trilinos_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/publicTrilinos
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
    RESULT_VARIABLE HAD_ERROR)
  
   message(STATUS "out: ${_out}")
   message(STATUS "err: ${_err}")
   message(STATUS "res: ${HAD_ERROR}")
   if(HAD_ERROR)
	message(FATAL_ERROR "Cannot clone Trilinos repository!")
   endif()
#else()
#  set(CTEST_CHECKOUT_COMMAND)
endif()

set(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")

# Get the SCOREC repo

if(NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/publicTrilinos/SCOREC")
#  EXECUTE_PROCESS(COMMAND "${CTEST_SVN_COMMAND}" 
#    checkout ${SCOREC_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/publicTrilinos/SCOREC
#    OUTPUT_VARIABLE _out
#    ERROR_VARIABLE _err
#    RESULT_VARIABLE HAD_ERROR)
  EXECUTE_PROCESS(COMMAND "${CTEST_GIT_COMMAND}" 
    clone ${SCOREC_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/publicTrilinos/SCOREC
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

# Get Albany

if(NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/Albany")
#  set(CTEST_CHECKOUT_COMMAND "${CTEST_GIT_COMMAND} clone ${Albany_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/Albany")
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

# Get Tpetra branch of Albany

if(NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/AlbanyT")
#  set(CTEST_CHECKOUT_COMMAND "${CTEST_GIT_COMMAND} clone ${Albany_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/Albany")
  EXECUTE_PROCESS(COMMAND "${CTEST_GIT_COMMAND}" 
    clone -b tpetra ${Albany_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/AlbanyT
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
    RESULT_VARIABLE HAD_ERROR)
  
   message(STATUS "out: ${_out}")
   message(STATUS "err: ${_err}")
   message(STATUS "res: ${HAD_ERROR}")
   if(HAD_ERROR)
	message(FATAL_ERROR "Cannot clone Albany repository, Tpetra branch!")
   endif()

endif()

ctest_start(${CTEST_TEST_TYPE})

# Send the project structure to CDash

IF(CTEST_DO_SUBMIT)
CTEST_SUBMIT(FILES "${CTEST_SCRIPT_DIRECTORY}/Project.xml"
          RETURN_VALUE  HAD_ERROR
            )

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot submit Albany Project.xml!")
endif()
ENDIF()

# Update Trilinos
SET_PROPERTY (GLOBAL PROPERTY SubProject Trilinos)
SET_PROPERTY (GLOBAL PROPERTY Label Trilinos)

ctest_update(SOURCE "${CTEST_SOURCE_DIRECTORY}/publicTrilinos" RETURN_VALUE count)
message("Found ${count} changed files")

IF(CTEST_DO_SUBMIT)
CTEST_SUBMIT(PARTS Update
          RETURN_VALUE  HAD_ERROR
            )

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot update Trilinos!")
endif()
ENDIF()

# Update the SCOREC repo
SET_PROPERTY (GLOBAL PROPERTY SubProject SCOREC)
SET_PROPERTY (GLOBAL PROPERTY Label SCOREC)

#set(CTEST_UPDATE_COMMAND "${CTEST_SVN_COMMAND}")
set(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
ctest_update(SOURCE "${CTEST_SOURCE_DIRECTORY}/publicTrilinos/SCOREC" RETURN_VALUE count)
message("Found ${count} changed files")

IF(CTEST_DO_SUBMIT)
CTEST_SUBMIT(PARTS Update
          RETURN_VALUE  HAD_ERROR
            )

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot update SCOREC!")
endif()
ENDIF()

# Update Albany
SET_PROPERTY (GLOBAL PROPERTY SubProject AlbanyMasterBranch)
SET_PROPERTY (GLOBAL PROPERTY Label AlbanyMasterBranch)

set(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
CTEST_UPDATE(SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany" RETURN_VALUE count)
message("Found ${count} changed files")

IF(CTEST_DO_SUBMIT)
CTEST_SUBMIT(PARTS Update
          RETURN_VALUE  HAD_ERROR
            )

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot update Albany!")
endif()
ENDIF()

# Update Albany Tpetra branch
SET_PROPERTY (GLOBAL PROPERTY SubProject AlbanyTpetraBranch)
SET_PROPERTY (GLOBAL PROPERTY Label AlbanyTpetraBranch)

set(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
CTEST_UPDATE(SOURCE "${CTEST_SOURCE_DIRECTORY}/AlbanyT" RETURN_VALUE count)
message("Found ${count} changed files")

IF(CTEST_DO_SUBMIT)
CTEST_SUBMIT(PARTS Update
          RETURN_VALUE  HAD_ERROR
            )

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot update Albany Tpetra branch!")
endif()
ENDIF()

# Configure the Trilinos/SCOREC build
IF (BUILD_TRI_SCOREC)
SET_PROPERTY (GLOBAL PROPERTY SubProject Trilinos)
SET_PROPERTY (GLOBAL PROPERTY Label Trilinos)

SET(CONFIGURE_OPTIONS
  "-Wno-dev"
  "-DTrilinos_EXTRA_REPOSITORIES:STRING=SCOREC"
  "-DTrilinos_ENABLE_SCOREC:BOOL=ON"
  "-DSCOREC_DISABLE_STRONG_WARNINGS:BOOL=ON"
  "-DCMAKE_BUILD_TYPE:STRING=NONE"
  "-DCMAKE_CXX_FLAGS:STRING=-O3 -w"
  "-DCMAKE_C_FLAGS:STRING=-O3 -w"
  "-DCMAKE_Fortran_FLAGS:STRING=-O3 -w"
#
  "-DTrilinos_ENABLE_ThyraTpetraAdapters:BOOL=ON"
  "-DTrilinos_ENABLE_Ifpack2:BOOL=ON"
  "-DTrilinos_ENABLE_Amesos2:BOOL=ON"
  "-DTrilinos_ENABLE_Zoltan2:BOOL=ON"
  "-DTrilinos_ENABLE_MueLu:BOOL=ON"
#
  "-DZoltan_ENABLE_ULONG_IDS:BOOL=ON"
  "-DTeuchos_ENABLE_COMPLEX:BOOL=OFF"
#
  "-DTPL_ENABLE_MPI:BOOL=ON"
  "-DMPI_BASE_DIR:PATH=${PREFIX_DIR}"
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
  "-DTPL_ENABLE_BoostAlbLib:BOOL=ON"
  "-DBoost_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DBoost_LIBRARY_DIRS:PATH=${PREFIX_DIR}/lib"
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
  "-DCMAKE_INSTALL_PREFIX:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
  "-DTrilinos_EXTRA_LINK_FLAGS='-L${PREFIX_DIR}/lib -lhdf5_hl -lhdf5 -lz -lm'"
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
  "-DTrilinos_ENABLE_MOOCHO:BOOL=ON"
  "-DTrilinos_ENABLE_OptiPack:BOOL=ON"
  "-DTrilinos_ENABLE_GlobiPack:BOOL=ON"
  "-DTrilinos_ENABLE_Stokhos:BOOL=ON"
  "-DTrilinos_ENABLE_Isorropia:BOOL=ON"
  "-DTrilinos_ENABLE_Piro:BOOL=ON"
  "-DTrilinos_ENABLE_STKClassic:BOOL=ON"
  "-DTrilinos_ENABLE_Teko:BOOL=ON"
  "-DTrilinos_ENABLE_Zoltan:BOOL=ON"
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
  )

# Turn off developer warnings
if(NOT EXISTS "${CTEST_BINARY_DIRECTORY}/TriBuild")
  FILE(MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/TriBuild)
endif()

CTEST_CONFIGURE(
          BUILD "${CTEST_BINARY_DIRECTORY}/TriBuild"
          SOURCE "${CTEST_SOURCE_DIRECTORY}/publicTrilinos"
          OPTIONS "${CONFIGURE_OPTIONS}"
          RETURN_VALUE HAD_ERROR
)

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot configure Trilinos/SCOREC build!")
endif()

IF(CTEST_DO_SUBMIT)
CTEST_SUBMIT(PARTS Configure
          RETURN_VALUE  HAD_ERROR
            )

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot submit Trilinos/SCOREC configure results!")
endif()
ENDIF()

# SCOREC build
SET_PROPERTY (GLOBAL PROPERTY SubProject SCOREC)
SET_PROPERTY (GLOBAL PROPERTY Label SCOREC)
SET(CTEST_BUILD_TARGET "SCOREC_libs")

MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

CTEST_BUILD(
          BUILD "${CTEST_BINARY_DIRECTORY}/TriBuild"
          RETURN_VALUE  HAD_ERROR
          NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
)

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot build Trilinos!")
endif()

IF(CTEST_DO_SUBMIT)
CTEST_SUBMIT(PARTS Build
          RETURN_VALUE  HAD_ERROR
            )

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot submit Trilinos/SCOREC build results!")
endif()
ENDIF()

# Trilinos
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

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot build Trilinos!")
endif()

IF(CTEST_DO_SUBMIT)
CTEST_SUBMIT(PARTS Build
          RETURN_VALUE  HAD_ERROR
            )

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot submit Trilinos/SCOREC build results!")
endif()
ENDIF()
ENDIF()


# Configure the Trilinos NewSTK build
IF (BUILD_TRI_NEW_STK)
SET_PROPERTY (GLOBAL PROPERTY SubProject TrilinosNewSTK)
SET_PROPERTY (GLOBAL PROPERTY Label TrilinosNewSTK)

SET(CONFIGURE_OPTIONS
  "-Wno-dev"
  "-DCMAKE_BUILD_TYPE:STRING=NONE"
  "-DCMAKE_CXX_FLAGS:STRING=-O3 -w"
  "-DCMAKE_C_FLAGS:STRING=-O3 -w"
  "-DCMAKE_Fortran_FLAGS:STRING=-O3 -w"
  "-DTPL_ENABLE_MPI:BOOL=ON"
  "-DMPI_BASE_DIR:PATH=${PREFIX_DIR}"
  "-DSEACAS_ENABLE_SEACASSVDI:BOOL=OFF"
  "-DTrilinos_ENABLE_SEACASFastq:BOOL=OFF"
  "-DTrilinos_ENABLE_SEACASBlot:BOOL=OFF"
  "-DTrilinos_ENABLE_SEACASPLT:BOOL=OFF"
  "-DTPL_ENABLE_X11:BOOL=OFF"
  "-DTrilinos_ENABLE_STK:BOOL=ON"
  "-DTrilinos_ENABLE_STKClassic:BOOL=OFF"
  "-DTrilinos_ENABLE_SEACASIoss:BOOL=ON"
  "-DTrilinos_ENABLE_SEACASExodus:BOOL=ON"
  "-DTrilinos_ENABLE_STKUtil:BOOL=ON"
  "-DTrilinos_ENABLE_STKTopology:BOOL=ON"
  "-DTrilinos_ENABLE_STKMesh:BOOL=ON"
  "-DTrilinos_ENABLE_STKIO:BOOL=ON"
  "-DTrilinos_ENABLE_STKSearch:BOOL=OFF"
  "-DTrilinos_ENABLE_STKSearchUtil:BOOL=OFF"
  "-DTrilinos_ENABLE_STKTransfer:BOOL=ON"
  "-DTrilinos_ENABLE_STKUnit_tests:BOOL=OFF"
  "-DTrilinos_ENABLE_STKDoc_tests:BOOL=OFF"
  "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF"
  "-DTrilinos_VERBOSE_CONFIGURE:BOOL=OFF"
  "-DBoost_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DBoostAlbLib_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DBoost_LIBRARY_DIRS:PATH=${PREFIX_DIR}/lib"
  "-DBoostAlbLib_LIBRARY_DIRS:PATH=${PREFIX_DIR}/lib"
  "-DTPL_ENABLE_Netcdf:STRING=ON"
  "-DNetcdf_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DNetcdf_LIBRARY_DIRS:PATH=${PREFIX_DIR}/lib"
  "-DTrilinos_EXTRA_LINK_FLAGS='-L${PREFIX_DIR}/lib -lhdf5_hl -lhdf5 -lz -lm'"
  "-DTPL_ENABLE_HDF5:STRING=ON"
  "-DHDF5_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DHDF5_LIBRARY_DIRS:PATH=${PREFIX_DIR}/lib"
  "-DTPL_ENABLE_Zlib:STRING=ON"
  "-DZlib_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DZlib_LIBRARY_DIRS:PATH=${PREFIX_DIR}/lib"
  "-DTPL_ENABLE_ParMETIS:STRING=ON"
  "-DParMETIS_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DParMETIS_LIBRARY_DIRS:PATH=${PREFIX_DIR}/lib"
  "-DTrilinos_ENABLE_ThyraTpetraAdapters:BOOL=ON"
  "-DTrilinos_ENABLE_Ifpack2:BOOL=ON"
  "-DTrilinos_ENABLE_Amesos2:BOOL=ON"
  "-DTrilinos_ENABLE_Zoltan2:BOOL=ON"
  "-DTrilinos_ENABLE_MueLu:BOOL=ON"
  "-DZoltan_ENABLE_ULONG_IDS:BOOL=ON"
  "-DTPL_ENABLE_SuperLU:STRING=ON"
  "-DSuperLU_INCLUDE_DIRS:PATH=${PREFIX_DIR}/SuperLU_4.3/include"
  "-DSuperLU_LIBRARY_DIRS:PATH=${PREFIX_DIR}/SuperLU_4.3/lib"
  "-DCMAKE_INSTALL_PREFIX:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstallNewSTK"
  "-DTrilinos_ASSERT_MISSING_PACKAGES:BOOL=OFF"
#
  "-DDART_TESTING_TIMEOUT:STRING=600"
  "-DTPL_ENABLE_Boost:BOOL=ON"
  "-DTPL_ENABLE_BoostAlbLib:BOOL=ON"
  "-DTrilinos_ENABLE_ThreadPool:BOOL=ON"
#
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
  "-DTrilinos_ENABLE_MOOCHO:BOOL=ON"
  "-DTrilinos_ENABLE_OptiPack:BOOL=ON"
  "-DTrilinos_ENABLE_GlobiPack:BOOL=ON"
  "-DTrilinos_ENABLE_Stokhos:BOOL=ON"
  "-DTrilinos_ENABLE_Isorropia:BOOL=ON"
  "-DTrilinos_ENABLE_Piro:BOOL=ON"
  "-DTrilinos_ENABLE_STKClassic:BOOL=ON"
  "-DTrilinos_ENABLE_Teko:BOOL=ON"
  "-DTrilinos_ENABLE_Zoltan:BOOL=ON"
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
  "-DTPL_ENABLE_Matio:BOOL=OFF"
  "-DTeuchos_ENABLE_COMPLEX:BOOL=OFF"
  "-DTrilinos_ENABLE_TESTS:BOOL=OFF"
  "-DTrilinos_ENABLE_TriKota:BOOL=OFF"
  "-DTrilinos_ENABLE_PyTrilinos:BOOL=OFF"
  )

# Turn off developer warnings
if(NOT EXISTS "${CTEST_BINARY_DIRECTORY}/TriBuildNewSTK")
  FILE(MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/TriBuildNewSTK)
endif()

CTEST_CONFIGURE(
          BUILD "${CTEST_BINARY_DIRECTORY}/TriBuildNewSTK"
          SOURCE "${CTEST_SOURCE_DIRECTORY}/publicTrilinos"
          OPTIONS "${CONFIGURE_OPTIONS}"
          RETURN_VALUE HAD_ERROR
)

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot configure TrilinosNewSTK build!")
endif()

IF(CTEST_DO_SUBMIT)
CTEST_SUBMIT(PARTS Configure
          RETURN_VALUE  HAD_ERROR
            )

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot submit TrilinosNewSTK configure results!")
endif()
ENDIF()

SET(CTEST_BUILD_TARGET install)

MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

CTEST_BUILD(
          BUILD "${CTEST_BINARY_DIRECTORY}/TriBuildNewSTK"
          RETURN_VALUE  HAD_ERROR
          NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
          APPEND
)

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot build TrilinosNewSTK!")
endif()

IF(CTEST_DO_SUBMIT)
CTEST_SUBMIT(PARTS Build
          RETURN_VALUE  HAD_ERROR
            )

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot submit TrilinosNewSTK build results!")
endif()
ENDIF()
ENDIF()

# Configure the Albany build (master branch without SCOREC tools)
# As folks move to github, its probably good to have a near full
# build of the master branch
IF (BUILD_ALB_BASE)
SET_PROPERTY (GLOBAL PROPERTY SubProject AlbanyMasterBranch)
SET_PROPERTY (GLOBAL PROPERTY Label AlbanyMasterBranch)

SET(CONFIGURE_OPTIONS
  "-DALBANY_TRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstallNewSTK"
  "-DENABLE_LCM:BOOL=ON"
  "-DENABLE_LCM_SPECULATIVE:BOOL=OFF"
  "-DENABLE_HYDRIDE:BOOL=ON"
  "-DENABLE_SCOREC:BOOL=OFF"
  "-DENABLE_SG_MP:BOOL=ON"
  "-DENABLE_FELIX:BOOL=ON"
  "-DENABLE_AERAS:BOOL=OFF"
  "-DENABLE_CHECK_FPE:BOOL=ON"
  )

if(NOT EXISTS "${CTEST_BINARY_DIRECTORY}/Albany")
  FILE(MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/Albany)
endif()

CTEST_CONFIGURE(
          BUILD "${CTEST_BINARY_DIRECTORY}/Albany"
          SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
          OPTIONS "${CONFIGURE_OPTIONS}"
          RETURN_VALUE HAD_ERROR
          APPEND
)

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot configure Albany build!")
endif()

IF(CTEST_DO_SUBMIT)
CTEST_SUBMIT(PARTS Configure
          RETURN_VALUE  HAD_ERROR
            )

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot submit Albany configure results!")
endif()
ENDIF()

# Build Albany

SET(CTEST_BUILD_TARGET all)

MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

CTEST_BUILD(
          BUILD "${CTEST_BINARY_DIRECTORY}/Albany"
          RETURN_VALUE  HAD_ERROR
          NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
          APPEND
)

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot build Albany!")
endif()

IF(CTEST_DO_SUBMIT)
CTEST_SUBMIT(PARTS Build
          RETURN_VALUE  HAD_ERROR
)

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot submit Albany build results!")
endif()
ENDIF()

# Run Albany tests

CTEST_TEST(
              BUILD "${CTEST_BINARY_DIRECTORY}/Albany"
#              PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
#              INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
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
ENDIF()

# Configure the Albany Tpetra branch build
IF (BUILD_ALB_TPETRA)
SET_PROPERTY (GLOBAL PROPERTY SubProject AlbanyTpetraBranch)
SET_PROPERTY (GLOBAL PROPERTY Label AlbanyTpetraBranch)

SET(CONFIGURE_OPTIONS
  "-DALBANY_TRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
  "-DENABLE_LCM:BOOL=ON"
  "-DENABLE_LCM_SPECULATIVE:BOOL=OFF"
  "-DENABLE_HYDRIDE:BOOL=ON"
  "-DENABLE_SCOREC:BOOL=ON"
  "-DENABLE_SG_MP:BOOL=OFF"
  "-DENABLE_QCAD:BOOL=OFF"
  "-DENABLE_MOR:BOOL=ON"
  "-DENABLE_CHECK_FPE:BOOL=ON"
  )

if(NOT EXISTS "${CTEST_BINARY_DIRECTORY}/AlbanyT")
  FILE(MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/AlbanyT)
endif()

CTEST_CONFIGURE(
          BUILD "${CTEST_BINARY_DIRECTORY}/AlbanyT"
          SOURCE "${CTEST_SOURCE_DIRECTORY}/AlbanyT"
          OPTIONS "${CONFIGURE_OPTIONS}"
          RETURN_VALUE HAD_ERROR
          APPEND
)

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot configure Albany Tpetra branch build!")
endif()

IF(CTEST_DO_SUBMIT)
CTEST_SUBMIT(PARTS Configure
          RETURN_VALUE  HAD_ERROR
            )

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot submit Albany Tpetra branch configure results!")
endif()
ENDIF()

# Build Albany Tpetra branch

SET(CTEST_BUILD_TARGET all)

MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

CTEST_BUILD(
          BUILD "${CTEST_BINARY_DIRECTORY}/AlbanyT"
          RETURN_VALUE  HAD_ERROR
          NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
          APPEND
)

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot build Albany Tpetra branch!")
endif()

IF(CTEST_DO_SUBMIT)
CTEST_SUBMIT(PARTS Build
          RETURN_VALUE  HAD_ERROR
            )

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot submit Albany Tpetra branch build results!")
endif()
ENDIF()

# Run Albany Tpetra branch tests

CTEST_TEST(
              BUILD "${CTEST_BINARY_DIRECTORY}/AlbanyT"
#              PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
#              INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
              #NUMBER_FAILED  TEST_NUM_FAILED
)

IF(CTEST_DO_SUBMIT)
CTEST_SUBMIT(PARTS Test
          RETURN_VALUE  HAD_ERROR
            )

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot submit Albany Tpetra branch test results!")
endif()
ENDIF()
ENDIF()

# Configure the Albany Tpetra branch build using GO = long long
IF (BUILD_ALB_TPETRA64)
SET_PROPERTY (GLOBAL PROPERTY SubProject AlbanyTpetra64Build)
SET_PROPERTY (GLOBAL PROPERTY Label AlbanyTpetra64Build)

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
  "-DENABLE_MOR:BOOL=ON"
  "-DENABLE_CHECK_FPE:BOOL=ON"
  )

if(NOT EXISTS "${CTEST_BINARY_DIRECTORY}/AlbanyT64")
  FILE(MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/AlbanyT64)
endif()

# The 64 bit build pulls code from the Tpetra branch - checked out in AlbanyT

CTEST_CONFIGURE(
          BUILD "${CTEST_BINARY_DIRECTORY}/AlbanyT64"
          SOURCE "${CTEST_SOURCE_DIRECTORY}/AlbanyT"
          OPTIONS "${CONFIGURE_OPTIONS}"
          RETURN_VALUE HAD_ERROR
          APPEND
)

# Read the CTestCustom.cmake file to turn off ignored tests

#CTEST_READ_CUSTOM_FILES("${CTEST_BINARY_DIRECTORY}/AlbanyT64")

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot configure Albany Tpetra 64 branch build!")
endif()

IF(CTEST_DO_SUBMIT)
CTEST_SUBMIT(PARTS Configure
          RETURN_VALUE  HAD_ERROR
            )

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot submit Albany Tpetra 64 branch configure results!")
endif()
ENDIF()

# Build Albany Tpetra 64 branch

SET(CTEST_BUILD_TARGET all)

MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

CTEST_BUILD(
          BUILD "${CTEST_BINARY_DIRECTORY}/AlbanyT64"
          RETURN_VALUE  HAD_ERROR
          NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
          APPEND
)

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot build Albany Tpetra 64 branch!")
endif()

IF(CTEST_DO_SUBMIT)
CTEST_SUBMIT(PARTS Build
          RETURN_VALUE  HAD_ERROR
            )

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot submit Albany Tpetra 64 branch build results!")
endif()
ENDIF()

# Run Albany Tpetra 64 branch tests

CTEST_TEST(
              BUILD "${CTEST_BINARY_DIRECTORY}/AlbanyT64"
#              PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
#              INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
              #NUMBER_FAILED  TEST_NUM_FAILED
)

IF(CTEST_DO_SUBMIT)
CTEST_SUBMIT(PARTS Test
          RETURN_VALUE  HAD_ERROR
            )

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot submit Albany Tpetra 64 branch test results!")
endif()
ENDIF()
ENDIF()

