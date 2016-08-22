#!/bin/bash

echo "----Shannon CUDA Trilinos build----" >& a
grep "Compiler errors" nightly_log_shannonTrilinos.txt >& b 
grep "Compiler warnings" nightly_log_shannonTrilinos.txt >& c 
echo "----Shannon CUDA Albany build----" >& d
grep "Compiler errors" nightly_log_shannonAlbany.txt >& e
grep "Compiler warnings" nightly_log_shannonAlbany.txt >& f
cat a b >& out 
rm a b 
cat out c >& out2 
rm out c 
cat out2 d >& out3 
rm out2 d 
cat out3 e >& out4 
rm out3 e 
cat out4 f >& test_summary.txt 
rm out4 f 
