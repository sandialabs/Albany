#!/bin/bash

# Crontab entry
#
# Run at midnight every day
#
# 00 00 * * * /fasttmp/ghansen/nightly/nightly_cron_script.sh

cd /fasttmp/dibanez/trilinos
source setup.sh

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=/fasttmp/dibanez/trilinos/nightly/log_$now
CMAKE_SCRIPT="/fasttmp/dibanez/trilinos/albany/doc/dashboards/avatar.scorec.rpi.edu/dan_nightly.cmake"

ctest -VV -S $CMAKE_SCRIPT 2>&1 > $LOG_FILE

