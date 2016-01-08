
#!/bin/bash

BASE_DIR=/project/projectdirs/piscees/nightlyEdisonCDash

/usr/bin/mail -s "Edison nightly test results" "ikalash@sandia.gov, agsalin@sandia.gov" -F "Irina Tezaur"  < $BASE_DIR/test_summary.txt
