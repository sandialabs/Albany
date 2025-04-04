
grep "Test   #" ctest_nightly_perf_tests.cmake >& results0
grep "Test  #" ctest_nightly_perf_tests.cmake >& results1
cat results0 results1 >& results11
grep "Test #" ctest_nightly_perf_tests.cmake >& results0
cat results11 results0 >& results1
echo "" >> results1
grep " tests failed" ctest_nightly_perf_tests.cmake >& results2
cat results1 results2 >& results3
grep "Total Test" ctest_nightly_perf_tests.cmake >& results4
cat results3 results4 >& results5
echo "" >> results5
grep "(Failed)" ctest_nightly_perf_tests.cmake >& results6
cat results5 results6 >& results7
grep "(Timeout)" ctest_nightly_perf_tests.cmake >& results8
cat results7 results8 >& results
echo "" >> results 
echo "The Albany CDash site can be accessed here: https://sems-cdash-son.sandia.gov/cdash/index.php?project=Albany" >> results
echo "" >> results
echo "The Jupyter notebooks containing Frontier ALI performance data can be accessed here: https://sandialabs.github.io/ali-perf-data/ali/index_frontier.html" >> results
echo "" >> results
rm results0 results1 results11 results2 results3 results4 results5 results6 results7 results8
#bash send_email.sh