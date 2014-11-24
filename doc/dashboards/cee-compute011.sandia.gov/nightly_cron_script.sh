#!/bin/sh

# Crontab entry
#
# Run at midnight every day
#
# 00 00 * * * /ascldap/users/gahanse/Codes/Albany/doc/dashboards/cee-compute011.sandia.gov/nightly_cron_script.sh

module purge
module load sierra-devel
module load sierra-mkl/15.0-2015.0.090
module unload sierra-git/1.7.3
module load sierra-git/2.0.0

cd /projects/AppComp/nightly/cee-compute011

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=/projects/AppComp/nightly/cee-compute011/nightly_$now

eval "env  TEST_DIRECTORY=/projects/AppComp/nightly/cee-compute011 SCRIPT_DIRECTORY=/ascldap/users/gahanse/Codes/Albany/doc/dashboards/cee-compute011.sandia.gov ctest -VV -S /ascldap/users/gahanse/Codes/Albany/doc/dashboards/cee-compute011.sandia.gov/ctest_nightly.cmake" >> $LOG_FILE 2>&1


