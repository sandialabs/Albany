
#!/bin/bash

#source $1 

#TTT=`grep "" /home/ikalash/nightlyCDash/nightly_log_32bit.txt`

#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "Edison nightly test results" "ikalash@sandia.gov" < /global/homes/i/ikalash/nightlyEdisonCDash/test_summary.txt

