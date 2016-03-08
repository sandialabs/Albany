#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /home/ikalash/nightlyCDash/nightly_log_cismAlbany.txt -c`
TTTT=`grep "(Not Run)" /home/ikalash/nightlyCDash/nightly_log_cismAlbany.txt -c`

#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "cism-piscees (felix_interface): $TTT tests failed, $TTTT tests not run" "ikalash@sandia.gov" < /home/ikalash/nightlyCDash/results_cismAlbany
/bin/mail -s "cism-piscees (felix_interface): $TTT tests failed, $TTTT tests not run" "agsalin@sandia.gov" < /home/ikalash/nightlyCDash/results_cismAlbany
