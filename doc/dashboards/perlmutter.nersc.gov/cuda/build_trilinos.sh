#!/bin/sh

BASE_DIR=/pscratch/sd/j/jwatkins/nightlyCDashPerlmutterCuda
DEPLOY_DIR=/global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/cuda

source ${BASE_DIR}/pm_gpu_gnu_modules.sh >& ${BASE_DIR}/modules_albany.out

bash ${BASE_DIR}/nightly_cron_script_trilinos_pm_gpu.sh

cp ${BASE_DIR}/biweekly_log_pm_gpu_Trilinos.txt ${DEPLOY_DIR}/logs/biweekly_log_pm_gpu_Trilinos.txt

chmod -R 2770 ${DEPLOY_DIR}
chown -R :fanssie ${DEPLOY_DIR}