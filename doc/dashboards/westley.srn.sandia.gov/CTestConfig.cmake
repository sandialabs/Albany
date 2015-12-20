## This file should be placed in the root directory of your project.
## Then modify the CMakeLists.txt file in the root directory of your
## project to incorporate the testing dashboard.
##
## # The following are required to submit to the CDash dashboard:
##   ENABLE_TESTING()
##   INCLUDE(CTest)

set(CTEST_PROJECT_NAME "Albany")
# Run test at/after 21:00 (9:00PM MDT --> 3:00 UTC, 8:00PM MST --> 3:00 UTC)
set(CTEST_NIGHTLY_START_TIME "03:00:00 UTC")

set(CTEST_DROP_SITE "westley.srn.sandia.gov")
set(CTEST_DROP_SITE "cdash.sandia.gov")
set(CTEST_DROP_LOCATION "/CDash-2-3-0/submit.php?project=Albany")
set(CTEST_DROP_SITE_CDASH TRUE)
