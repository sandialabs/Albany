
cd /home/ikalash/nightlySpackBuild
grep "Test   #" spack_ctest.out >& results0
grep "Test  #" spack_ctest.out >& results1 
cat results0 results1 >& results11
grep "Test #" spack_ctest.out >& results0
cat results11 results0 >& results1
grep " tests failed" spack_ctest.out >& results2 
cat results1 results2 >& results3
grep "Total Test" spack_ctest.out >& results4
cat results3 results4 >& results5
grep "(Failed)" spack_ctest.out >& results6 
cat results5 results6 >& results_spack
echo "" >> results_spack 
rm results0 results1 results11 results2 results3 results4 results5 results6
