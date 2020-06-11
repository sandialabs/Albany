
cd /nightlyCDash
grep "Test   #" nightly_logT.txt >& results0
grep "Test  #" nightly_logT.txt >& results1 
cat results0 results1 >& results11
grep "Test #" nightly_logT.txt >& results0
cat results11 results0 >& results1
grep " tests failed" nightly_logT.txt >& results2 
cat results1 results2 >& results3
grep "Total Test" nightly_logT.txt >& results4
cat results3 results4 >& results5
grep "(Failed)" nightly_logT.txt >& results6 
cat results5 results6 >& resultsT
echo "" >> resultsT 
echo "The Albany CDash site can be accessed here: https://sems-cdash-son.sandia.gov/cdash/index.php?project=Albany" >> resultsT 
echo "" >> resultsT 
rm results0 results1 results11 results2 results3 results4 results5 results6
