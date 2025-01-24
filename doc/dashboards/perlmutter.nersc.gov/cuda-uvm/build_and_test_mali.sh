#!/bin/bash -login

#SBATCH -A m4274_g
#SBATCH --job-name=AlbanyTesting
#SBATCH --output=logs/AlbanyTesting.gpu.mali.%j.out
#SBATCH --error=logs/AlbanyTesting.gpu.mali.%j.err
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH --gpu-bind=none

BASE_DIR=/pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-cuda-uvm
DEPLOY_DIR=/global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/cuda-uvm

source ${BASE_DIR}/pm_gpu_gnu_modules.sh >& ${BASE_DIR}/modules_albany.out

bash ${BASE_DIR}/nightly_cron_script_mali_pm_gpu.sh

mkdir ${BASE_DIR}/repos/compass_tests
cp ${BASE_DIR}/CTestTestfile.cmake ${BASE_DIR}/repos/compass_tests
cp ${BASE_DIR}/test_and_log.sh ${BASE_DIR}/repos/compass_tests
bash ${BASE_DIR}/nightly_cron_script_compass_pm_gpu.sh

cp biweekly_log_pm_gpu_Compass.txt ${DEPLOY_DIR}/logs/biweekly_log_pm_gpu_Compass.txt
cp biweekly_log_pm_gpu_MALI.txt ${DEPLOY_DIR}/logs/biweekly_log_pm_gpu_MALI.txt

chmod -R 2770 ${DEPLOY_DIR}
chown -R :fanssie ${DEPLOY_DIR}