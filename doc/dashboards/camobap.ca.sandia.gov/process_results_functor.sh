
cd /home/ikalash/Desktop/nightlyCDash
grep "Test   #" nighty_log_functor.txt >& results0
grep "Test  #" nighty_log_functor.txt >& results1 
cat results0 results1 >& results11
grep "Test #" nighty_log_functor.txt >& results0
cat results11 results0 >& results1
grep " tests failed" nighty_log_functor.txt >& results2 
cat results1 results2 >& results3
grep "Total Test" nighty_log_functor.txt >& results4
cat results3 results4 >& results5
grep "(Failed)" nighty_log_functor.txt >& results6 
cat results5 results6 >& results_functor
rm results0 results1 results11 results2 results3 results4 results5 results6
