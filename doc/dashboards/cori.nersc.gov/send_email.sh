
#!/bin/bash

BASE_DIR=/project/projectdirs/piscees/nightlyCoriCDash

/usr/bin/mail -s "Cori nightly test results" "ikalash@sandia.gov, agsalin@sandia.gov" -F "Irina Tezaur"  < $BASE_DIR/test_summary.txt
