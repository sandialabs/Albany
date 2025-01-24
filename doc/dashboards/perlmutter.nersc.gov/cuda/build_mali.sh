#!/bin/csh

BASE_DIR=/pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-cuda
DEPLOY_DIR=/global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/cuda

source ${BASE_DIR}/pm_gpu_gnu_modules.sh >& ${BASE_DIR}/modules_albany.out

bash ${BASE_DIR}/nightly_cron_script_mali_pm_gpu.sh

mkdir ${BASE_DIR}/repos/compass_tests
cp ${BASE_DIR}/CTestTestfile.cmake ${BASE_DIR}/repos/compass_tests
cp ${BASE_DIR}/test_and_log.sh ${BASE_DIR}/repos/compass_tests
bash ${BASE_DIR}/nightly_cron_script_compass_pm_gpu.sh

cp biweekly_log_pm_gpu_Compass.txt ${DEPLOY_DIR}/logs/biweekly_log_pm_cpu_Compass.txt
cp biweekly_log_pm_gpu_MALI.txt ${DEPLOY_DIR}/logs/biweekly_log_pm_cpu_MALI.txt

chmod -R 2770 ${DEPLOY_DIR}
chown -R :fanssie ${DEPLOY_DIR}