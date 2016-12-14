
cd /home/ikalash/nightlyCDash
grep "Test   #" nightly_log_32bit.txt >& results0
grep "Test  #" nightly_log_32bit.txt >& results1 
cat results0 results1 >& results11
grep "Test #" nightly_log_32bit.txt >& results0
cat results11 results0 >& results1
grep " tests failed" nightly_log_32bit.txt >& results2 
cat results1 results2 >& results3
grep "Total Test" nightly_log_32bit.txt >& results4
cat results3 results4 >& results5
grep "(Failed)" nightly_log_32bit.txt >& results6 
cat results5 results6 >& results_32bit
echo "" >> results_32bit 
echo "The Albany CDash site can be accessed here: http://cdash.sandia.gov/CDash-2-3-0/index.php?project=Albany" >> results_32bit 
echo "" >> results_32bit 
rm results0 results1 results11 results2 results3 results4 results5 results6
