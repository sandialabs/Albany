#!/bin/sh

# Crontab entry
#
# Run at midnight every day
#
# 00 00 * * * /fasttmp/ghansen/nightly/nightly_cron_script.sh

PATH=/users/ghansen/Trilinos/MPI_REL/bin:/users/ghansen/bin:/fasttmp/ghansen/Trilinos/MPI_RELD/bin:/usr/local/bin:/usr/bin:/bin:/usr/X11R6/bin
export PATH

LD_LIBRARY_PATH=/users/ghansen/lib:/users/ghansen/lib64:/fasttmp/ghansen/netcdf-4.2.1.1/install/lib:/fasttmp/ghansen/hdf5-1.8.9/hdf5/lib:/usr/local/parasolid/25.1.181/shared_object
export LD_LIBRARY_PATH

cd /fasttmp/ghansen/nightly

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=/fasttmp/ghansen/nightly/nightly_$now

eval "env  TEST_DIRECTORY=/fasttmp/ghansen/nightly ctest -VV -S /fasttmp/ghansen/Albany/doc/dashboards/avatar.scorec.rpi.edu/ctest_nightly_mpi_opt_avatar.cmake" >> $LOG_FILE 2>&1

