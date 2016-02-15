
#!/bin/bash

BASE_DIR=/project/projectdirs/piscees/nightlyGaiaCDash

/usr/bin/mail -s "Gaia nightly test results" "wfspotz@sandia.gov ikalash@sandia.gov, agsalin@sandia.gov" -F "Bill Spotz"  < $BASE_DIR/test_summary.txt
