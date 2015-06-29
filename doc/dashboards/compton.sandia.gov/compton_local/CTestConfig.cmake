## This file should be placed in the root directory of your project.
## Then modify the CMakeLists.txt file in the root directory of your
## project to incorporate the testing dashboard.
##
## # The following are required to submit to the CDash dashboard:
##   ENABLE_TESTING()
##   INCLUDE(CTest)

set(CTEST_PROJECT_NAME "Albany")
set(CTEST_NIGHTLY_START_TIME "00:00:00 MDT")

set(CTEST_DROP_SITE "compton.sandia.gov")
set(CTEST_DROP_LOCATION "nightly/Albany")
set(CTEST_DROP_METHOD "cp")
set(CTEST_TRIGGER_SITE "")
set(CTEST_DROP_SITE_USER "")
