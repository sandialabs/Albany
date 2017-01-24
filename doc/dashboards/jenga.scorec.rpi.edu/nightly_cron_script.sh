#!/bin/bash

# Crontab entry
#
# Run at midnight every day
#
# 00 00 * * * /users/mperego/nightly-tests-scripts/nightly_cron_script.sh

export PATH=/users/ghansen/Trilinos/MPI_REL/bin:/users/ghansen/bin:/usr/local/bin:/usr/bin:/bin:/usr/X11R6/bin

export LD_LIBRARY_PATH=/users/ghansen/lib:/users/ghansen/lib64:/users/ghansen/ompi-gcc/lib:/usr/local/intel/11.1/069/mkl/lib/em64t

cd /users/mperego/nightly

LOG_FILE=/users/mperego/nightly/nightly_log.txt
if [ -f $LOG_FILE ]; then
  rm $LOG_FILE
fi

eval "env  TEST_DIRECTORY=/users/mperego/nightly SCRIPT_DIRECTORY=/users/mperego ctest -VV -S /users/mperego/nightly-tests-scripts/ctest_nightly_jenga.cmake" >> $LOG_FILE 2>&1


