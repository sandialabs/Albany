#!/bin/bash

echo "----Cori Trilinos build----" >& a
grep "Compiler errors" nightly_log_coriTrilinos.txt >& b 
grep "Compiler warnings" nightly_log_coriTrilinos.txt >& c 
echo "----Cori Albany build----" >& d
grep "Compiler errors" nightly_log_coriAlbany.txt >& e
grep "Compiler warnings" nightly_log_coriAlbany.txt >& f
echo "----Cori CISM-Albany build----" >& g
grep "Compiler errors" nightly_log_coriCismAlbany.txt >& h
grep "Compiler warnings" nightly_log_coriCismAlbany.txt >& i
cat a b >& out 
rm a b 
cat out c >& out2 
rm out c 
cat out2 d >& out3 
rm out2 d 
cat out3 e >& out4 
rm out3 e 
cat out4 f >& out5 
rm out4 f 
cat out5 g >& out6 
rm out5 g 
cat out6 h >& out7 
rm out6 h 
cat out7 i >& test_summary.txt
echo "" >> test_summary.txt
echo "The Cori CDash site can be accessed here: http://my.cdash.org/index.php?project=Albany" >> test_summary.txt
echo "" >> test_summary.txt
rm out7 i 
