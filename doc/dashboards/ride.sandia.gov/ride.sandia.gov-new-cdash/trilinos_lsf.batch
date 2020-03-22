#!/bin/bash                                           
#BSUB -a "openmpi"                                    
#BSUB -n 32
#BSUB -W 02:00                                                           
#BSUB -R "span[ptile=16]"                             
#BSUB -o trilinosCUDA.out                                 
#BSUB -e trilinosCUDA.err                                 

cd /ascldap/users/ikalash/nightlyCDash 

rm -rf repos
rm -rf build
rm -rf ctest_nightly.cmake.work
rm -rf nightly_log*
rm -rf results*
rm -rf slurm* 
rm -rf modules*out 

cat trilinos ctest_nightly.cmake.frag >& ctest_nightly.cmake  
bash -c -l "source ride_modules_cuda.sh >& modules_trilinos.out; bash nightly_cron_script_trilinos_ride.sh" 
