#!/bin/bash

#source $1 

TESTDIR=/home/lcm/LCM/nightlyCDash

TTT=`grep "(Failed)" $TESTDIR/nightly_log_LCM_release.txt -c`
TTTT=`grep "(Not Run)" $TESTDIR/nightly_log_LCM_release.txt -c`
TTTTT=`grep "(Timeout)" $TESTDIR/nightly_log_LCM_release.txt -c`

/bin/mail -r `whoami`@sandia.gov -s "Albany LCM (master, gcc-release, algol): $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, albany-regression@software.sandia.gov" < $TESTDIR/results_LCM_release
