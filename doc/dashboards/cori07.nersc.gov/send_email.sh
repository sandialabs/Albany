
#!/bin/bash

BASE_DIR=/project/projectdirs/piscees/nightlyCoriCDash

/usr/bin/mail -s "Cori nightly test results" "ikalash@sandia.gov" -F "Irina Tezaur"  < $BASE_DIR/results_coriCismAlbany
