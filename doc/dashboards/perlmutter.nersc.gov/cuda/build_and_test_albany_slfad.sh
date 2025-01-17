#!/bin/bash -login

#SBATCH -A m4274_g
#SBATCH --job-name=AlbanyTesting
#SBATCH --output=logs/AlbanyTesting.gpu.sfad12.%j.out
#SBATCH --error=logs/AlbanyTesting.gpu.sfad12.%j.err
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH --gpu-bind=none

BASE_DIR=/pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-cuda
DEPLOY_DIR=/global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/cuda

source ${BASE_DIR}/pm_gpu_gnu_modules.sh >& ${BASE_DIR}/modules_albany.out
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${DEPLOY_DIR}/builds/TrilinosInstall/lib64"

bash ${BASE_DIR}/nightly_cron_script_albany_pm_gpu.sh slfad none
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${DEPLOY_DIR}/builds/AlbanyInstall/lib64"

cp biweekly_log_pm_gpu_Albany_slfad.txt ${DEPLOY_DIR}/logs/biweekly_log_pm_gpu_Albany_slfad.txt

chmod -R 2770 ${DEPLOY_DIR}
chown -R :fanssie ${DEPLOY_DIR}