
cd /home/ikalash/LCM
grep "Test   #" albany-serial-intel-release.log >& results0
grep "Test  #" albany-serial-intel-release.log >& results1 
cat results0 results1 >& results11
grep "Test #" albany-serial-intel-release.log >& results0
cat results11 results0 >& results1
grep " tests failed" albany-serial-intel-release.log >& results2 
cat results1 results2 >& results3
grep "Total Test" albany-serial-intel-release.log >& results4
cat results3 results4 >& results5
echo "" >> results5
grep "(Failed)" albany-serial-intel-release.log >& results6 
cat results5 results6 >& results7
grep "\*Timeout" albany-serial-intel-release.log >& results8
cat results7 results8 >& results
echo "" >> results
echo "The Albany CDash site can be accessed here: https://sems-cdash-son.sandia.gov/cdash/index.php?project=Albany" >> results
echo "" >> results
rm results0 results1 results11 results2 results3 results4 results5 results6 results7 results8
