#!/bin/bash                                           

#rm -rf repos
#rm -rf build
#rm -rf ctest_nightly.cmake.work
#rm -rf nightly_log*
#rm -rf results*
#rm -rf slurm* 
#rm -rf modules*out 

ulimit -c 0


bash -c -l "source waterman_modules_cuda.sh >& modules_albany.out; bash nightly_cron_script_trilinos_waterman.sh; bash nightly_cron_script_albany_waterman.sh; bash nightly_cron_script_albany_waterman_sfad.sh sfad4; bash nightly_cron_script_albany_waterman_sfad.sh sfad6; bash nightly_cron_script_albany_waterman_sfad.sh sfad8; bash nightly_cron_script_albany_waterman_sfad.sh sfad12"
bash process_results_ctest.sh
#bash send_email_ctest.sh  

