# This script expects to be called as follows:
# cmake -P lcm_build.cmake
#       -DSCRIPT_NAME="config-build-test.sh"
#       -DPACKAGE="trilinos"
#       -DBUILD_THREADS="16"

include("${CMAKE_CURRENT_LIST_DIR}/lcm_do_package.cmake")

set(CTEST_TEST_TYPE Nightly)
set(CTEST_CMAKE_GENERATOR  "Unix Makefiles")
set(CTEST_PROJECT_NAME "LCM")
set (CTEST_COMMAND "ctest -D ${CTEST_TEST_TYPE}")

message("SCRIPT_NAME ${SCRIPT_NAME}")
message("PACKAGE ${PACKAGE}")
message("BUILD_THREADS ${BUILD_THREADS}")

set(BUILD_ID_STRING "$ENV{ARCH}-$ENV{TOOL_CHAIN}-$ENV{BUILD_TYPE}")
message("BUILD_ID_STRING ${BUILD_ID_STRING}")

set(PASS_ARGS "RESULT_VARIABLE" "PACKAGE_ERR")
if (SCRIPT_NAME MATCHES "clean")
  set(PASS_ARGS ${PASS_ARGS} "CLEAN_BUILD" "CLEAN_INSTALL")
endif()
if (SCRIPT_NAME MATCHES "update")
  set(PASS_ARGS ${PASS_ARGS} "DO_UPDATE")
endif()
if (SCRIPT_NAME MATCHES "config")
  set(PASS_ARGS ${PASS_ARGS} "DO_CONFIG")
endif()
if (SCRIPT_NAME MATCHES "build")
  set(PASS_ARGS ${PASS_ARGS} "DO_BUILD")
endif()
if (SCRIPT_NAME MATCHES "test")
  set(PASS_ARGS ${PASS_ARGS} "DO_TEST")
endif()
if (SCRIPT_NAME MATCHES "dash")
  set(CTEST_DO_SUBMIT ON)
else()
  set(CTEST_DO_SUBMIT OFF)
endif()
set(PASS_ARGS ${PASS_ARGS} "PACKAGE" "${PACKAGE}")
set(PASS_ARGS ${PASS_ARGS} "BUILD_THREADS" "${BUILD_THREADS}")
set(PASS_ARGS ${PASS_ARGS} "BUILD_ID_STRING" "${BUILD_ID_STRING}")

#cmake_host_system_information(RESULT LCM_HOSTNAME QUERY HOSTNAME)
set(LCM_HOSTNAME "skybridge-login5")

message("LCM_HOSTNAME ${LCM_HOSTNAME}")

set(CTEST_BUILD_NAME "${LCM_HOSTNAME}-${PACKAGE}")
set(CTEST_SITE "${LCM_HOSTNAME}")
set(CTEST_SOURCE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")
set(CTEST_BINARY_DIRECTORY "$ENV{LCM_DIR}/${PACKAGE}-build-${BUILD_ID_STRING}")
message("CTEST_BINARY_DIRECTORY ${CTEST_BINARY_DIRECTORY}")
snl_mkdir("${CTEST_BINARY_DIRECTORY}")

set(PASS_ARGS ${PASS_ARGS})

configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/CTestCustom.cmake"
  "${CTEST_BINARY_DIRECTORY}/CTestCustom.cmake"
  COPYONLY)

ctest_start(${CTEST_TEST_TYPE})

lcm_do_package(${PASS_ARGS})

if (PACKAGE_ERR)
  message(FATAL_ERROR "lcm_do_package returned \"${PACKAGE_ERR}\"")
else()
  message("lcm_do_package returned \"${PACKAGE_ERR}\"")
endif()
