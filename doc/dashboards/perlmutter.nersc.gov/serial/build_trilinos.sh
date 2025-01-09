#!/bin/sh

BASE_DIR=/pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-serial
DEPLOY_DIR=/global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/serial

source ${BASE_DIR}/pm_cpu_gnu_modules.sh >& ${BASE_DIR}/modules_albany.out

bash ${BASE_DIR}/nightly_cron_script_trilinos_pm_cpu.sh

cp ${BASE_DIR}/biweekly_log_pm_cpu_Trilinos.txt ${DEPLOY_DIR}/logs/biweekly_log_pm_cpu_Trilinos.txt

chmod -R 2770 ${DEPLOY_DIR}
chown -R :fanssie ${DEPLOY_DIR}
