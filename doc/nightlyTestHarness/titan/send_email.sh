/bin/mail -s "TITAN! Albany ($ALBANY_BRANCH): $TTT" "agsalin@sandia.gov" < $ALBOUTDIR/albany_make.out
#!/bin/bash -l

TTT=`grep "Albany" $ALBOUTDIR/albany_make.out | tail -1`

# "albany-regression@software.sandia.gov"
/bin/mail -s "Albany ($ALBANY_BRANCH) on Titan: $TTT" "agsalin@sandia.gov" < $ALBOUTDIR/albany_make.out

