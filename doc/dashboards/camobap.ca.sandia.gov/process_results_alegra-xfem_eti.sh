
cd /nightlyCDash
grep "Test #" nightly_log_alegra-xfem_eti.txt >& results_alegra-xfem_eti
echo "" >> results_alegra-xfem_eti 
echo "The CDash site where test results are posted can be accessed here: https://sems-cdash-son.sandia.gov/cdash/index.php?project=Albany" >> results_alegra-xfem_eti 
echo "" >> results_alegra-xfem_eti
