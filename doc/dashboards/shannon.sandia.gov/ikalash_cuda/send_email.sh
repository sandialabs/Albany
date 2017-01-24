
#!/bin/bash

BASE_DIR=/home/ikalash/nightlyCDash

mail -s "Shannon CUDA nightly test results" "ikalash@sandia.gov, agsalin@sandia.gov, mperego@sandia.gov, jwatkin@sandia.gov" -F "Irina Tezaur"  < $BASE_DIR/test_summary.txt
