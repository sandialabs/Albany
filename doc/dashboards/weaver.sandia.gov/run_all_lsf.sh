#!/bin/bash                                           

#rm -rf repos
#rm -rf build
#rm -rf ctest_nightly.cmake.work
#rm -rf nightly_log*
#rm -rf results*
#rm -rf slurm* 
#rm -rf modules*out 

ulimit -c 0


#bash -c -l "source weaver_modules_cuda.sh >& modules_albany.out; bash nightly_cron_script_albany_weaver.sh; bash nightly_cron_script_albany_weaver_sfad.sh sfad6; bash nightly_cron_script_albany_weaver_sfad.sh sfad12"
bash -c -l "source weaver_modules_cuda.sh >& modules_albany.out; bash nightly_cron_script_trilinos_weaver.sh; bash nightly_cron_script_albany_weaver.sh; bash nightly_cron_script_albany_weaver_sfad.sh sfad6; bash nightly_cron_script_albany_weaver_sfad.sh sfad12; bash nightly_cron_script_albany_weaver_sfad.sh sfad16; bash nightly_cron_script_albany_weaver_sfad.sh sfad24 bash nightly_cron_script_mali_weaver.sh"
bash process_results_ctest.sh
#bash send_email_ctest.sh  

