cmake_minimum_required(VERSION 2.8)

# Begin User inputs:
set( CTEST_SITE             "avatar.scorec.rpi.edu" ) # generally the output of hostname
#set( CTEST_DASHBOARD_ROOT   "$ENV{TEST_DIRECTORY}" ) # writable path
set( CTEST_DASHBOARD_ROOT   "/fasttmp/ghansen/nightly" ) # writable path
set( CTEST_CMAKE_GENERATOR  "Unix Makefiles" ) # What is your compilation apps ?
set( CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?

set( CTEST_PROJECT_NAME         "Albany" )
set( CTEST_SOURCE_NAME          repos)
set( CTEST_BUILD_NAME           "linux-gcc-${CTEST_BUILD_CONFIGURATION}")
set( CTEST_BINARY_NAME          build)

SET(PREFIX_DIR /users/ghansen)

SET (CTEST_SOURCE_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_SOURCE_NAME}")
SET (CTEST_BINARY_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_BINARY_NAME}")

configure_file(${CTEST_SCRIPT_DIRECTORY}/CTestConfig.cmake
               ${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake COPYONLY)

# Must match what is in CDash project 'Trilinos'
#SET(CTEST_NIGHTLY_START_TIME "00:00:00 UTC")
#SET(CTEST_TEST_TYPE Nightly)
SET(CTEST_TEST_TYPE Experimental)
SET (CTEST_CMAKE_COMMAND "${PREFIX_DIR}/bin/cmake")
SET (CTEST_COMMAND "${PREFIX_DIR}/bin/ctest -D ${CTEST_TEST_TYPE}")
#SET (CMAKE_MAKE_PROGRAM "/usr/bin/make -j 8")
SET (CTEST_BUILD_FLAGS -j8)

# Set actual CTest/CDash settings
#set(CTEST_CONFIGURE_COMMAND "${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE:STRING=${CTEST_BUILD_CONFIGURATION}")
#set(CTEST_CONFIGURE_COMMAND "${CTEST_CONFIGURE_COMMAND} -DWITH_TESTING:BOOL=ON ${CTEST_BUILD_OPTIONS}")
#set(CTEST_CONFIGURE_COMMAND "${CTEST_CONFIGURE_COMMAND} \"-G${CTEST_CMAKE_GENERATOR}\"")
#set(CTEST_CONFIGURE_COMMAND "${CTEST_CONFIGURE_COMMAND} \"${CTEST_SOURCE_DIRECTORY}\"")

#IF (NOT DEFINED CTEST_DROP_METHOD)
#  SET_DEFAULT_AND_FROM_ENV(CTEST_DROP_METHOD "http")
#ENDIF()
  SET(CTEST_DROP_METHOD "http")

IF (CTEST_DROP_METHOD STREQUAL "http")
#  SET_DEFAULT_AND_FROM_ENV(CTEST_DROP_SITE "dummy.com")
#  SET_DEFAULT_AND_FROM_ENV(CTEST_PROJECT_NAME "MockProjectName")
#  SET_DEFAULT_AND_FROM_ENV(CTEST_DROP_LOCATION "/cdash/submit.php?project=MockProjectName")
#  SET_DEFAULT_AND_FROM_ENV(CTEST_TRIGGER_SITE "")
#  SET_DEFAULT_AND_FROM_ENV(CTEST_DROP_SITE_CDASH TRUE)
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
SET(SCOREC_REPOSITORY_LOCATION https://redmine.scorec.rpi.edu/svn/buildutil/trunk/cmake)
#SET(Albany_REPOSITORY_LOCATION ghansen@jumpgate.scorec.rpi.edu:/users/ghansen/Albany.git)
SET(Albany_REPOSITORY_LOCATION git@github.com:gahansen/Albany.git)

# Initial cache info
set( CACHE_CONTENTS "
SITE:STRING=${CTEST_SITE}
CMAKE_BUILD_TYPE:STRING=Release
CMAKE_GENERATOR:INTERNAL=${CTEST_CMAKE_GENERATOR}
BUILD_TESTING:BOOL=OFF
PRODUCT_REPO:STRING=${Albany_REPOSITORY_LOCATION}
" )
#CMAKE_MAKE_PROGRAM:STRING=/usr/local/bin/gmake -i -j 8

ctest_empty_binary_directory( "${CTEST_BINARY_DIRECTORY}" )
file(WRITE "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt" "${CACHE_CONTENTS}")

#configure_file(${CTEST_SCRIPT_DIRECTORY}/CMakeLists.txt
#               ${CTEST_SOURCE_DIRECTORY}/CMakeLists.txt COPYONLY)


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
  EXECUTE_PROCESS(COMMAND "${CTEST_SVN_COMMAND}" 
    checkout ${SCOREC_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/publicTrilinos/SCOREC
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

ctest_start(${CTEST_TEST_TYPE})
ctest_update(SOURCE "${CTEST_SOURCE_DIRECTORY}/publicTrilinos" RETURN_VALUE res)
if(res)
	message(FATAL_ERROR "Cannot update Trilinos repository!")
endif()

# Update the SCOREC repo

set(CTEST_UPDATE_COMMAND "${CTEST_SVN_COMMAND}")
ctest_update(SOURCE "${CTEST_SOURCE_DIRECTORY}/publicTrilinos/SCOREC" RETURN_VALUE res)
if(res)
	message(FATAL_ERROR "Cannot update Scorec repository!")
endif()

# Update Albany

set(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
ctest_update(SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany" RETURN_VALUE res)
if(res)
	message(FATAL_ERROR "Cannot update Albany repository!")
endif()

# Configure the Trilinos/SCOREC build

# Note - we explicity change the Trilinos build from RELEASE to specifying the build flags to turn off
# warnings - while refactoring is underway

SET(CONFIGURE_OPTIONS
  "-DTrilinos_EXTRA_REPOSITORIES:STRING=SCOREC"
  "-DTrilinos_CONFIGURE_OPTIONS_FILE:FILEPATH=${CTEST_SOURCE_DIRECTORY}/publicTrilinos/sampleScripts/AlbanySettings.cmake"
  "-DCMAKE_BUILD_TYPE:STRING=NONE"
  "-DCMAKE_CXX_FLAGS:STRING=-O3 -w"
  "-DCMAKE_C_FLAGS:STRING=-O3 -w"
  "-DCMAKE_Fortran_FLAGS:STRING=-O3 -w"
  "-DTPL_ENABLE_MPI:BOOL=ON"
  "-DMPI_BASE_DIR:PATH=${PREFIX_DIR}"
  "-DTPL_ENABLE_Matio:BOOL=OFF"
  "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF"
  "-DTrilinos_VERBOSE_CONFIGURE:BOOL=OFF"
  "-DBoost_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DBoostAlbLib_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
  "-DBoost_LIBRARY_DIRS:PATH=${PREFIX_DIR}/lib"
  "-DBoostAlbLib_LIBRARY_DIRS:PATH=${PREFIX_DIR}/lib"
  "-DNetcdf_INCLUDE_DIRS:PATH=${PREFIX_DIR}/parallel/include"
  "-DNetcdf_LIBRARY_DIRS:PATH=${PREFIX_DIR}/parallel/lib"
  "-DHDF5_INCLUDE_DIRS:PATH=${PREFIX_DIR}/parallel/include"
  "-DHDF5_LIBRARY_DIRS:PATH=${PREFIX_DIR}/parallel/lib"
  "-DParMETIS_INCLUDE_DIRS:PATH=${PREFIX_DIR}/parallel/include"
  "-DParMETIS_LIBRARY_DIRS:PATH=${PREFIX_DIR}/parallel/lib"
  "-DTrilinos_ENABLE_SCOREC:BOOL=ON"
  "-DTrilinos_ENABLE_SCORECmeshadapt:BOOL=ON"
  "-DTrilinos_ENABLE_SCORECpumi_geom_parasolid:BOOL=ON"
  "-DTPL_ENABLE_Parasolid:BOOL=ON"
  "-DParasolid_INCLUDE_DIRS:PATH=/usr/local/parasolid/25.1.181"
  "-DParasolid_LIBRARY_DIRS:PATH=/usr/local/parasolid/25.1.181/shared_object"
  "-DCMAKE_INSTALL_PREFIX:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
  "-DTrilinos_ASSERT_MISSING_PACKAGES:BOOL=OFF"
  )

CTEST_CONFIGURE(
          BUILD "${CTEST_BINARY_DIRECTORY}"
          SOURCE "${CTEST_SOURCE_DIRECTORY}/publicTrilinos"
          OPTIONS "${CONFIGURE_OPTIONS}"
          RETURN_VALUE HAD_ERROR
)

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot configure Trilinos build!")
endif()

CTEST_BUILD(
          BUILD "${CTEST_BINARY_DIRECTORY}"
          RETURN_VALUE  HAD_ERROR
          NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
          APPEND
)

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot build Trilinos!")
endif()

#execute_process(COMMAND "/usr/bin/make" "install" 
execute_process(COMMAND "${CMAKE_MAKE_PROGRAM}" "install" 
  WORKING_DIRECTORY ${CTEST_BINARY_DIRECTORY} 
  RESULT_VARIABLE makeInstallResult 
  OUTPUT_VARIABLE makeInstallLog 
  ERROR_VARIABLE makeInstallLog
)

file(WRITE ${CTEST_BINARY_DIRECTORY}/makeinstall.log
  "${makeInstallLog}")

# Configure the Albany build

SET(CONFIGURE_OPTIONS
  "-DALBANY_TRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
  "-DENABLE_LCM:BOOL=ON"
  "-DENABLE_LCM_SPECULATIVE:BOOL=OFF"
  "-DENABLE_HYDRIDE:BOOL=ON"
  "-DENABLE_SCOREC:BOOL=ON"
  "-DENABLE_SG_MP:BOOL=ON"
  )

if(NOT EXISTS "${CTEST_BINARY_DIRECTORY}/Albany")
  FILE(MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/Albany)
endif()

CTEST_CONFIGURE(
          BUILD "${CTEST_BINARY_DIRECTORY}/Albany"
          SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
          OPTIONS "${CONFIGURE_OPTIONS}"
          RETURN_VALUE HAD_ERROR
)

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot configure Albany build!")
endif()

CTEST_BUILD(
          BUILD "${CTEST_BINARY_DIRECTORY}/Albany"
          RETURN_VALUE  HAD_ERROR
          NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
          APPEND
)

if(HAD_ERROR)
	message(FATAL_ERROR "Cannot build Albany!")
endif()

CTEST_TEST(
              BUILD "${CTEST_BINARY_DIRECTORY}/Albany"
#              PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
#              INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
              #NUMBER_FAILED  TEST_NUM_FAILED
)

#set (CTEST_SOURCE_DIRECTORY
#> "/var/opt/buildtools/release/VEHICLES_2_1_0/CAR_2_1/export")
#> set (CTEST_BINARY_DIRECTORY
#> "/var/opt/buildtools/release/VEHICLES_2_1_0/CAR_2_1/build.Release" )
#> set(CTEST_PROJECT_NAME "VEHICLES_2_1_0")
#> set(CTEST_NIGHTLY_START_TIME "00:00:00 UTC")
#> set(CTEST_DROP_METHOD "http")
#> set(CTEST_DROP_SITE "localhost")
#> set(CTEST_DROP_LOCATION "/CDash/submit.php?project=VEHICLES_2_1_0")
#> set(CTEST_DROP_SITE_CDASH TRUE)
#> SET(CTEST_CMAKE_GENERATOR "Unix Makefiles")
#> SET(CTEST_BUILD_COMMAND make)
#> file(WRITE "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt" "
#> SITE:STRING=${hostname}
#> CMAKE_BUILD_TYPE:STRING=Release
#> PRODUCT_DIR:STRING=/var/opt/buildtools/release/VEHICLES_2_1_0
#> ")
#> set(CTEST_SITE "localhost")
#> set_property(GLOBAL PROPERTY SubProject "CAR_2_1")
#> set_property(GLOBAL PROPERTY Label "CAR_2_1")
#> set (CTEST_BUILD_TARGET "CAR_2_1")
#> CTEST_START("Experimental")
#> CTEST_CONFIGURE(BUILD "${CTEST_BINARY_DIRECTORY}" RETURN_VALUE configRes)
#> if (${configRes} EQUAL 0)
#>    CTEST_BUILD(BUILD "${CTEST_BINARY_DIRECTORY}" RETURN_VALUE buildRes)
#> endif (${configRes} EQUAL 0)
#> if (${buildRes} EQUAL 0)
#>    execute_process(COMMAND "/usr/bin/make" "install" WORKING_DIRECTORY
#> ${CTEST_BINARY_DIRECTORY} RESULT_VARIABLE makeInstallResult OUTPUT_VARIABLE
#> makeInstallLog ERROR_VARIABLE makeInstallLog)
#>    file(WRITE ${CTEST_BINARY_DIRECTORY}/makeinstall.log
#> "${makeInstallLog}")
#>    CTEST_TEST(BUILD "${CTEST_BINARY_DIRECTORY}")
#> endif (${buildRes} EQUAL 0)
#> CTEST_SUBMIT(FILES
#> "/var/opt/buildtools/release/VEHICLES_2_1_0/Project.xml")
#> CTEST_SUBMIT()

## Pull in Albany options
#INCLUDE("${TRILINOS_BOOTSTRAP}/sampleScripts/AlbanySettings.cmake")
#
## Point at the public Repo
#SET(Trilinos_REPOSITORY_LOCATION https://software.sandia.gov/trilinos/repositories/publicTrilinos)
#
## Pull in Extra repos
#INCLUDE("${CTEST_SCRIPT_DIRECTORY}/ExtraRepositoriesList.cmake")
#
## CTEST_SCRIPT_DIRECTORY is the directory where this script is in
#INCLUDE("${CTEST_SCRIPT_DIRECTORY}/TrilinosCTestDriverCore.avatar.gcc.cmake")
#
##
## Set the options specific to this build case
##
#
#SET(BUILD_TYPE RELEASE)
#SET(BUILD_DIR_NAME MPI_OPT_DEV)
#SET(CTEST_TEST_TYPE Nightly)
##SET(CTEST_TEST_TIMEOUT 900)
#
#SET(PREFIX_DIR /users/ghansen)
#
#SET( EXTRA_CONFIGURE_OPTIONS
#  "-DTPL_ENABLE_MPI:BOOL=ON"
#  "-DMPI_BASE_DIR:PATH=${PREFIX_DIR}"
#  "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF"
#  "-DTrilinos_VERBOSE_CONFIGURE:BOOL=OFF"
#  "-DBoost_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
#  "-DBoostAlbLib_INCLUDE_DIRS:PATH=${PREFIX_DIR}/include"
#  "-DBoost_LIBRARY_DIRS:PATH=${PREFIX_DIR}/lib"
#  "-DBoostAlbLib_LIBRARY_DIRS:PATH=${PREFIX_DIR}/lib"
#  "-DNetcdf_INCLUDE_DIRS:PATH=${PREFIX_DIR}/parallel/include"
#  "-DNetcdf_LIBRARY_DIRS:PATH=${PREFIX_DIR}/parallel/lib"
#  "-DHDF5_INCLUDE_DIRS:PATH=${PREFIX_DIR}/parallel/include"
#  "-DHDF5_LIBRARY_DIRS:PATH=${PREFIX_DIR}/parallel/lib"
#  "-DParMETIS_INCLUDE_DIRS:PATH=${PREFIX_DIR}/parallel/include"
#  "-DParMETIS_LIBRARY_DIRS:PATH=${PREFIX_DIR}/parallel/lib"
#  "-DTrilinos_ASSERT_MISSING_PACKAGES:BOOL=OFF"
#  )
#
#
#SET(Trilinos_ASSERT_MISSING_PACKAGES FALSE)
#SET(Trilinos_IGNORE_PACKAGE_EXISTS_CHECK TRUE)
#
#
##
## Set the rest of the system-specific options and run the dashboard build/test
##
#
#TRILINOS_SYSTEM_SPECIFIC_CTEST_DRIVER()

#1084     CTEST_START(${CTEST_TEST_TYPE})
#1104     CTEST_UPDATE_WRAPPER( SOURCE "${CTEST_SOURCE_DIRECTORY}"
#      RETURN_VALUE  UPDATE_RETURN_VAL)
#
#1379        CTEST_CONFIGURE(
#          BUILD "${CTEST_BINARY_DIRECTORY}"
#          OPTIONS "${CONFIGURE_OPTIONS}" # New option!
#          RETURN_VALUE CONFIGURE_RETURN_VAL
#          )
#1420          CTEST_SUBMIT( PARTS configure notes )
#1436        CTEST_BUILD(
#          BUILD "${CTEST_BINARY_DIRECTORY}"
#          RETURN_VALUE  BUILD_LIBS_RETURN_VAL
#          NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
#          APPEND
#          )
#1459           CTEST_SUBMIT( PARTS build )
#1500             CTEST_TEST(
#              BUILD "${CTEST_BINARY_DIRECTORY}"
#              PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
#              INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
#              #NUMBER_FAILED  TEST_NUM_FAILED
#              )
#
#


