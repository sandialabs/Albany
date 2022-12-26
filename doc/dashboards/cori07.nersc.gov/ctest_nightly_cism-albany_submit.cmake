cmake_minimum_required (VERSION 2.8)
set (CTEST_DO_SUBMIT ON)
set (CTEST_TEST_TYPE Nightly)

# Begin User inputs:
set (CTEST_SITE "cori07.nersc.gov" ) # generally the output of hostname
set (CTEST_DASHBOARD_ROOT "$ENV{TEST_DIRECTORY}" ) # writable path
set (CTEST_SCRIPT_DIRECTORY "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set (CTEST_CMAKE_GENERATOR "Unix Makefiles" ) # What is your compilation apps ?
set (CTEST_CONFIGURATION  Release) # What type of build do you want ?

set (INITIAL_LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})

set (CTEST_PROJECT_NAME "Albany" )
set (CTEST_SOURCE_NAME repos)
#set (CTEST_BUILD_NAME "cori-CISM-Albany")
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

set (CTEST_NIGHTLY_START_TIME "01:00:00 UTC")
set (CTEST_CMAKE_COMMAND "${PREFIX_DIR}/bin/cmake")
set (CTEST_COMMAND "${PREFIX_DIR}/bin/ctest -D ${CTEST_TEST_TYPE}")
set (CTEST_FLAGS "-j16")
set (CTEST_BUILD_FLAGS "-j16")

set (CTEST_DROP_METHOD "https")

execute_process(COMMAND grep "Build name" nightly_log_coriCismAlbanyBuild.txt 
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		RESULT_VARIABLE BUILD_NAME_RESULT
		OUTPUT_FILE "build_name.txt")
execute_process(COMMAND bash get_build_name.sh 
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		RESULT_VARIABLE BUILD_NAME_RESULT
		OUTPUT_VARIABLE BUILD_NAME
		OUTPUT_STRIP_TRAILING_WHITESPACE)
message("IKT build_name = " ${BUILD_NAME})
set (CTEST_BUILD_NAME "${BUILD_NAME}")

find_program (CTEST_GIT_COMMAND NAMES git)
find_program (CTEST_SVN_COMMAND NAMES svn)

set (Albany_REPOSITORY_LOCATION git@github.com:sandialabs/Albany.git)
set (Trilinos_REPOSITORY_LOCATION git@github.com:trilinos/Trilinos.git)
set (cism-piscees_REPOSITORY_LOCATION  git@github.com:E3SM-Project/cism-piscees.git)


set (BOOST_DIR /project/projectdirs/piscees/tpl/boost_1_55_0) 
#set (NETCDF_DIR /opt/cray/pe/netcdf-hdf5parallel/4.4.0/GNU/5.1) 

ctest_start(${CTEST_TEST_TYPE})

ctest_submit (FILES "/project/projectdirs/piscees/nightlyCoriCDash/build/Testing/20221226-0100/Test.xml" RETURN_VALUE  S_HAD_ERROR)

if (S_HAD_ERROR)
  message ("Cannot submit Albany test results!")
endif ()
