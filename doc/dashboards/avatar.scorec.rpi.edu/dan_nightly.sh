#!/bin/bash

# Crontab entry
#
# Run at 3 am every day
#
# 00 03 * * * /fasttmp/dibanez/trilinos/albany/doc/dashboards/avatar.scorec.rpi.edu/dan_nightly.sh

cd /fasttmp/dibanez/cdash/trilinos

export PATH=/usr/local/cmake/latest/bin:$PATH
export PATH=/users/ghansen/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/ghansen/lib:/users/ghansen/lib64

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=/fasttmp/dibanez/cdash/trilinos/log_$now
CMAKE_SCRIPT="/fasttmp/dibanez/trilinos/albany/doc/dashboards/avatar.scorec.rpi.edu/dan_nightly.cmake"

ctest --debug -S $CMAKE_SCRIPT 2>&1 > $LOG_FILE

rm -rf build repos
