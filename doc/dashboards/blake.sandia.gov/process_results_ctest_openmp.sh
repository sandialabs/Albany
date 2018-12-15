
grep "Test   #" nightly_log_blakeAlbanyOpenMP.txt >& results0
grep "Test  #" nightly_log_blakeAlbanyOpenMP.txt >& results1
cat results0 results1 >& results11
grep "Test #" nightly_log_blakeAlbanyOpenMP.txt >& results0
cat results11 results0 >& results1
echo "" >> results1
grep " tests failed" nightly_log_blakeAlbanyOpenMP.txt >& results2
cat results1 results2 >& results3
grep "Total Test" nightly_log_blakeAlbanyOpenMP.txt >& results4
cat results3 results4 >& results5
grep "(Failed)" nightly_log_blakeAlbanyOpenMP.txt >& results6
cat results5 results6 >& results_openmp
echo "" >> results_openmp 
echo "The Albany CDash site can be accessed here: https://my.cdash.org/index.php?project=Albany" >> results_openmp
echo "" >> results_openmp
rm results0 results1 results11 results2 results3 results4 results5 results6
bash send_email_openmp.sh
