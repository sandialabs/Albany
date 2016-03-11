#!/bin/bash

echo "----Gaia Trilinos build----" >& a
grep "Compiler errors" nightly_log_gaiaTrilinos.txt >& b 
grep "Compiler warnings" nightly_log_gaiaTrilinos.txt >& c 
echo "----Gaia Albany build----" >& d
grep "Compiler errors" nightly_log_gaiaAlbany.txt >& e
grep "Compiler warnings" nightly_log_gaiaAlbany.txt >& f
cat a b >& out 
rm a b 
cat out c >& out2 
rm out c 
cat out2 d >& out3 
rm out2 d 
cat out3 e >& out4 
rm out3 e 
cat out4 f >& test_summary-txt
