#!/bin/sh

now=$(date +"%m_%d_%Y")
LOG_FILE=/fasttmp/ghansen/nightly/nightly_$now

eval "env  TEST_DIRECTORY=/fasttmp/ghansen/nightly ctest -S /fasttmp/ghansen/Albany/doc/dashboards/avatar.scorec.rpi.edu/ctest_nightly_mpi_opt_avatar.cmake" >> $LOG_FILE 2>&1

#eval "env  TEST_DIRECTORY=/fasttmp/ghansen/nightly ctest -VV -S /fasttmp/ghansen/Albany/doc/dashboards/avatar.scorec.rpi.edu/ctest_nightly_mpi_opt_avatar.cmake" >> $LOG_FILE 2>&1

