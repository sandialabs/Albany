#cmake_minimum_required (VERSION 2.8)
set (CTEST_DO_SUBMIT ON)
set (CTEST_TEST_TYPE Nightly)

# What to build and test
set (BUILD_ALBANY TRUE) 
set (ALBANY_RUNTESTS TRUE)

# Begin User inputs:
set (CTEST_SITE "mayer.sandia.gov" ) # generally the output of hostname
set (CTEST_DASHBOARD_ROOT "$ENV{TEST_DIRECTORY}" ) # writable path
set (CTEST_SCRIPT_DIRECTORY "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set (CTEST_CMAKE_GENERATOR "Unix Makefiles" ) # What is your compilation apps ?
set (CTEST_CONFIGURATION  Release) # What type of build do you want ?

set (INITIAL_LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})

set (CTEST_PROJECT_NAME "Albany" )
set (CTEST_SOURCE_NAME repos)
set (CTEST_BUILD_CONFIGURATION Release)
set (CTEST_BUILD_NAME "mayer-${CTEST_BUILD_CONFIGURATION}-Albany")
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
set (CTEST_CMAKE_COMMAND "cmake")
set (CTEST_COMMAND "ctest -D ${CTEST_TEST_TYPE}")
set (CTEST_FLAGS "-j8")
SET (CTEST_BUILD_FLAGS "-j8")

ctest_start(${CTEST_TEST_TYPE})

ctest_submit (FILES "/mscratch/albany/mayer/nightlyCDashJenkins/build/Testing/XXX-0100/Configure.xml" RETURN_VALUE  S_HAD_ERROR)

if (S_HAD_ERROR)
  message ("Cannot submit Albany configure results!")
endif ()

ctest_submit (FILES "/mscratch/albany/mayer/nightlyCDashJenkins/build/Testing/XXX-0100/Build.xml" RETURN_VALUE  S_HAD_ERROR)

if (S_HAD_ERROR)
  message ("Cannot submit Albany configure results!")
endif ()

ctest_submit (FILES "/mscratch/albany/mayer/nightlyCDashJenkins/build/Testing/XXX-0100/Test.xml" RETURN_VALUE  S_HAD_ERROR)

if (S_HAD_ERROR)
  message ("Cannot submit Albany configure results!")
endif ()
