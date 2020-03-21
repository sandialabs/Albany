#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /home/ikalash/nightlyCDashNewCDash/nightly_logFPE.txt -c`
TTTT=`grep "(Not Run)" /home/ikalash/nightlyCDashNewCDash/nightly_logFPE.txt -c`
TTTTT=`grep "(Timeout)" /home/ikalash/nightlyCDashNewCDash/nightly_logFPE.txt -c`
TT=`grep "...   Passed" /home/ikalash/nightlyCDashNewCDash/nightly_logFPE.txt -c`

#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "IKTAlbanyFPECheckDbg, camobap.ca.sandia.gov: $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, daibane@sandia.gov, tjfulle@sandia.gov, mperego@sandia.gov, lbertag@sandia.gov" < /home/ikalash/nightlyCDashNewCDash/resultsFPE
