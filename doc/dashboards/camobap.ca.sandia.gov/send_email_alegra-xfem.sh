#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /nightlyCDash/nightly_log_alegra-xfem.txt -c`
TTTT=`grep "(Not Run)" /nightlyCDash/nightly_log_alegra-xfem.txt -c`
TTTTT=`grep "(Timeout)" /nightlyCDash/nightly_log_alegra-xfem.txt -c`
TT=`grep "...   Passed" /nightlyCDash/nightly_log_alegra-xfem.txt -c`

echo "Subject: alegra-xfem, camobap.ca.sandia.gov: $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" >& a
echo "" >& b
cat a b >& c
cat c results_alegra-xfem >& d
mv d results_alegra-xfem
rm a b c
cat results_alegra-xfem | /usr/lib/sendmail -F ikalash@camobap.ca.sandia.gov -t "ikalash@sandia.gov"
#sendmail -s "alegra-xfem, camobap.ca.sandia.gov: $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov" < /nightlyCDash/results_alegra-xfem
