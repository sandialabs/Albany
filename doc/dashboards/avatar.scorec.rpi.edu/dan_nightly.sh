#!/bin/bash

# Crontab entry
#
# Run at 3 am every day
#
# 00 03 * * * /fasttmp/dibanez/trilinos/albany/doc/dashboards/avatar.scorec.rpi.edu/dan_nightly.sh

cd /fasttmp/dibanez/cdash/trilinos

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=/fasttmp/dibanez/cdash/trilinos/log_$now
CMAKE_SCRIPT="/fasttmp/dibanez/trilinos/albany/doc/dashboards/avatar.scorec.rpi.edu/dan_nightly.cmake"

ctest -VV -S $CMAKE_SCRIPT 2>&1 > $LOG_FILE

