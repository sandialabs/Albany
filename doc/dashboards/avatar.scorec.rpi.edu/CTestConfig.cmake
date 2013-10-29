## This file should be placed in the root directory of your project.
## Then modify the CMakeLists.txt file in the root directory of your
## project to incorporate the testing dashboard.
## # The following are required to uses Dart and the Cdash dashboard
##   ENABLE_TESTING()
##   INCLUDE(CTest)
set(CTEST_PROJECT_NAME "AlbanyRPI")
set(CTEST_NIGHTLY_START_TIME "01:00:00 UTC")

set(CTEST_DROP_METHOD "http")
#set(CTEST_DROP_SITE "cdash.example.com")
set(CTEST_DROP_SITE "")
#set(CTEST_DROP_LOCATION "/cdash/submit.php?project=AlbanyRPI")
set(CTEST_DROP_LOCATION "")
#set(CTEST_DROP_SITE_CDASH TRUE)
set(CTEST_DROP_SITE_CDASH FALSE)
