#!/bin/bash
#
# Analyze the Albany build on CDash for a given date.
#
# Usage:
#
#    albany_cdash_status.sh [--date=<YYYY-MM-DD>] \
#      --email-from-address=<email-from-address> \
#      --send-email-to=<email-1>,<email-2>,...
#
# Run with --help for more details.
#
# To use this, you first must clone TriBITS under this directory with:
#
#   git clone git@github.com:TriBITSPub/TriBITS.git
#
# or set the env var TRIBITS_DIR to the location of some recent copy of
# TriBITS (like in Trilinos 'develop') with:
#
#   env TRIBITS_DIR=<trilinos-src-dir>/cmake/tribits \
#   albany_cdash_status.sh [--date=<YYYY-MM-DD>] \
#      --email-from-address=<email-from-address> \
#      --send-email-to=<email-1>,<email-2>,...
#

# Find directory paths


_ABS_FILE_PATH=`readlink -f $0` || \
echo "Could not follow symlink to set ALBANY_CDASH_STATUS_DIR!"
if [ "$_ABS_FILE_PATH" != "" ] ; then
  ALBANY_CDASH_STATUS_DIR=`dirname $_ABS_FILE_PATH`
fi

if [ "$TRIBITS_DIR" == "" ] ; then
  TRIBITS_DIR="${ALBANY_CDASH_STATUS_DIR}/TriBITS/tribits"
fi

# Run the script

${TRIBITS_DIR}/ci_support/cdash_analyze_and_report.py \
--cdash-project-testing-day-start-time="01:00" \
--cdash-project-name="Albany" \
--build-set-name="Albany Nightly Builds" \
--cdash-site-url="https://sems-cdash-son.sandia.gov/cdash/" \
--cdash-builds-filters="filtercount=1&showfilters=1&field1=groupname&compare1=61&value1=Nightly" \
--cdash-nonpassed-tests-filters="filtercount=2&showfilters=1&filtercombine=and&field1=groupname&compare1=61&value1=Nightly&field2=status&compare2=62&value2=passed" \
--require-test-history-match-nonpassing-tests=off \
--limit-table-rows=50 \
--write-failing-tests-without-issue-trackers-to-file="albanyNightlyBuildsTwoif.csv" \
--write-email-to-file="albanyNightlyBuilds.html" \
--expected-builds-file="AlbanyExpectedBuildsWeekend.csv" \
"$@"
