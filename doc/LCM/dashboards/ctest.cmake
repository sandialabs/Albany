cmake_minimum_required (VERSION 2.8)
set (CTEST_DO_SUBMIT ON)
set (CTEST_TEST_TYPE lcm_test_type)

set (CTEST_SITE "lcm_test_site")
set (CTEST_SOURCE_NAME "lcm_source_dir")
set (CTEST_BINARY_NAME "lcm_build_dir")
set (CTEST_SUBPROJECT "lcm_project_xml")
set (CTEST_BUILD_NAME "lcm_build_name")

set (CTEST_SOURCE_DIRECTORY "${CTEST_SOURCE_NAME}")
set (CTEST_BINARY_DIRECTORY "${CTEST_BINARY_NAME}")

set (CTEST_DROP_METHOD "http")

if (CTEST_DROP_METHOD STREQUAL "http")
  set (CTEST_DROP_SITE "cdash.sandia.gov")
  set (CTEST_PROJECT_NAME "Albany")
  set (CTEST_DROP_LOCATION "/CDash-2-3-0/submit.php?project=Albany")
  set (CTEST_TRIGGER_SITE "")
  set (CTEST_DROP_SITE_CDASH TRUE)
endif ()

ctest_start(${CTEST_TEST_TYPE})

if (CTEST_DO_SUBMIT)
  ctest_submit (FILES "${CTEST_SUBPROJECT}"
    RETURN_VALUE  HAD_ERROR
    )

  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot submit Albany Project.xml!")
  endif ()
endif ()
