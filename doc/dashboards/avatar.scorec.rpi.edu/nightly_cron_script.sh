#!/bin/sh

# Crontab entry
#
# Run at midnight every day
#
# 00 00 * * * /lore/ghansen/nightly/nightly_cron_script.sh

PATH=/users/ghansen/Trilinos/MPI_REL/bin:/users/ghansen/bin:/usr/local/bin:/usr/bin:/bin:/usr/X11R6/bin
export PATH

LD_LIBRARY_PATH=/usr/local/intel/11.1/069/mkl/lib/em64t:/users/ghansen/lib:/users/ghansen/lib64:/usr/local/parasolid/25.1.181/shared_object
export LD_LIBRARY_PATH

cd /lore/ghansen/nightly

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=/lore/ghansen/nightly/nightly_$now

eval "env  TEST_DIRECTORY=/lore/ghansen/nightly SCRIPT_DIRECTORY=/users/ghansen/Albany/doc/dashboards/avatar.scorec.rpi.edu ctest -VV -S /users/ghansen/Albany/doc/dashboards/avatar.scorec.rpi.edu/ctest_nightly_mpi_opt_avatar.cmake" >> $LOG_FILE 2>&1


