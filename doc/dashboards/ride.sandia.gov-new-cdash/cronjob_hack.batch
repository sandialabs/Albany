#!/bin/bash                                           
#BSUB -a "openmpi"                                    
#BSUB -n 32
#BSUB -W 08:00                                                           
#BSUB -R "span[ptile=16]"                             
#BSUB -o albanyCDash.out                                 
#BSUB -e albanyCDash.err                                 

cd /ascldap/users/ikalash/nightlyCDash 

rm -rf repos
rm -rf build
rm -rf ctest_nightly.cmake.work
rm -rf nightly_log*
rm -rf results*
rm -rf slurm* 
rm -rf modules*out 

ulimit -c 0

cat trilinos ctest_nightly.cmake.frag >& ctest_nightly.cmake  
bash -c -l "source ride_modules_cuda.sh >& modules_trilinos.out; bash nightly_cron_script_trilinos_ride.sh"
cat albany ctest_nightly.cmake.frag >& ctest_nightly.cmake
bash -c -l "source ride_modules_cuda.sh >& modules_albany.out; bash nightly_cron_script_albany_ride.sh"
scp nightly_log_rideAlbany.txt ikalash@mockba.ca.sandia.gov:rideCDash/Albany
bash process_results_ctest.sh 
scp results_cuda ikalash@mockba.ca.sandia.gov:rideCDash/Albany
bash send_email_ctest.sh  

# recursively resubmit this script
X=$( date -d "1 days 9 am" "+%m:%d:%H:%M" )
bsub -b $X -q rhel7G < /ascldap/users/ikalash/nightlyCDash/cronjob_hack.batch
