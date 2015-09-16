## This file should be placed in the root directory of your project.
## Then modify the CMakeLists.txt file in the root directory of your
## project to incorporate the testing dashboard.
##
## # The following are required to submit to the CDash dashboard:
##   ENABLE_TESTING()
##   INCLUDE(CTest)

set(CTEST_PROJECT_NAME "Albany")
# Roughly midnite - 0700 UTC
set(CTEST_NIGHTLY_START_TIME "04:00:00 UTC")

#set(CTEST_DROP_METHOD "http")
#set(CTEST_DROP_SITE "cdash.sandia.gov")
#set(CTEST_DROP_LOCATION "/CDash-2-3-0/submit.php?project=Albany")
#set(CTEST_DROP_SITE_CDASH TRUE)

set(CTEST_DROP_SITE "shannon.sandia.gov")
set(CTEST_DROP_LOCATION "nightly/Albany")
set(CTEST_DROP_METHOD "cp")
set(CTEST_TRIGGER_SITE "")
set(CTEST_DROP_SITE_USER "")
