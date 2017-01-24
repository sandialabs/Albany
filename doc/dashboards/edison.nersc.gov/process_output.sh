#!/bin/bash

echo "----Edison Trilinos build----" >& a
grep "Compiler errors" nightly_log_edisonTrilinos.txt >& b 
grep "Compiler warnings" nightly_log_edisonTrilinos.txt >& c 
echo "----Edison Albany build----" >& d
grep "Compiler errors" nightly_log_edisonAlbany.txt >& e
grep "Compiler warnings" nightly_log_edisonAlbany.txt >& f
echo "----Edison CISM-Albany build----" >& g
grep "Compiler errors" nightly_log_edisonCismAlbany.txt >& h
grep "Compiler warnings" nightly_log_edisonCismAlbany.txt >& i
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
echo "The Edison CDash site can be accessed here: http://my.cdash.org/index.php?project=Albany" >> test_summary.txt
echo "" >> test_summary.txt
rm out7 i 
