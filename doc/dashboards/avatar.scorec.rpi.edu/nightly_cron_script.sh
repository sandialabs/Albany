#!/bin/bash

# Crontab entry
#
# Run at midnight every day
#
# 00 00 * * * /lore/ghansen/nightly/nightly_cron_script.sh

export PATH=/users/ghansen/Trilinos/MPI_REL/bin:/users/ghansen/bin:/usr/local/bin:/usr/bin:/bin:/usr/X11R6/bin

export LD_LIBRARY_PATH=/users/ghansen/lib64:/users/ghansen/ompi-gcc/lib:/usr/local/intel/11.1/069/mkl/lib/em64t

cd /lore/ghansen/nightly

LOG_FILE=/lore/ghansen/nightly/nightly_log.txt
if [ -f $LOG_FILE ]; then
  rm $LOG_FILE
fi

eval "env  TEST_DIRECTORY=/lore/ghansen/nightly SCRIPT_DIRECTORY=/users/ghansen/Albany/doc/dashboards/avatar.scorec.rpi.edu ctest -VV -S /users/ghansen/Albany/doc/dashboards/avatar.scorec.rpi.edu/ctest_nightly_avatar.cmake" >> $LOG_FILE 2>&1


