#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /nightlyCDash/nightly_log_alegra-xfem_eti.txt -c`
TTTT=`grep "(Not Run)" /nightlyCDash/nightly_log_alegra-xfem_eti.txt -c`
TTTTT=`grep "(Timeout)" /nightlyCDash/nightly_log_alegra-xfem_eti.txt -c`
TT=`grep "...   Passed" /nightlyCDash/nightly_log_alegra-xfem_eti.txt -c`

echo "Subject: alegra-xfem-eti, camobap.ca.sandia.gov: $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" >& a
echo "" >& b
cat a b >& c
cat c results_alegra-xfem_eti >& d
mv d results_alegra-xfem_eti
rm a b c
cat results_alegra-xfem_eti | /usr/lib/sendmail -F ikalash@camobap.ca.sandia.gov -t "ikalash@sandia.gov"
sendmail -s "alegra-xfem_eti, camobap.ca.sandia.gov: $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov" < /nightlyCDash/results_alegra-xfem_eti
