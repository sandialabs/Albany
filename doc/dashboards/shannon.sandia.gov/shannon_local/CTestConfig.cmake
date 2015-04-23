## This file should be placed in the root directory of your project.
## Then modify the CMakeLists.txt file in the root directory of your
## project to incorporate the testing dashboard.
##
## # The following are required to submit to the CDash dashboard:
##   ENABLE_TESTING()
##   INCLUDE(CTest)

set(CTEST_PROJECT_NAME "Albany")
# Roughly midnite - 0700 UTC
set(CTEST_NIGHTLY_START_TIME "07:00:00 UTC")

#set(CTEST_DROP_METHOD "http")
#set(CTEST_DROP_SITE "cdash.sandia.gov")
#set(CTEST_DROP_LOCATION "/CDash-2-3-0/submit.php?project=Albany")
#set(CTEST_DROP_SITE_CDASH TRUE)

set(CTEST_DROP_SITE "software-login.sandia.gov")
set(CTEST_DROP_LOCATION "Albany")
set(CTEST_DROP_METHOD "scp")
find_program(CTEST_SCP_COMMAND scp DOC "scp command for local copy of results")
set(CTEST_TRIGGER_SITE "")
set(CTEST_DROP_SITE_USER "")
# CTest does "scp file ${CTEST_DROP_SITE}:${CTEST_DROP_LOCATION}" so for
# local copy w/o needing sshd on localhost we arrange to have : in the
# absolute filepath
if (NOT EXISTS "${CTEST_DROP_SITE}:${CTEST_DROP_LOCATION}")
    message(FATAL_ERROR
      "must set ${CTEST_DROP_SITE}:${CTEST_DROP_LOCATION} to an existing directory")
endif (NOT EXISTS "${CTEST_DROP_SITE}:${CTEST_DROP_LOCATION}")
