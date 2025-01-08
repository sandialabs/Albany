#!/bin/sh

BASE_DIR=/pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-serial

source ${BASE_DIR}/pm_cpu_gnu_modules.sh >& ${BASE_DIR}/modules_albany.out

bash ${BASE_DIR}/nightly_cron_script_trilinos_pm_cpu.sh

cp biweekly_log_pm_cpu_Trilinos.txt /global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/serial/logs/biweekly_log_pm_cpu_Trilinos.txt

chmod -R 2770 /global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/serial
chown -R :fanssie /global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/serial