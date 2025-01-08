#!/bin/bash -login

#SBATCH -A m4274
#SBATCH --job-name=AlbanyTesting
#SBATCH --output=AlbanyTesting.cpu.sfad12.%j.out
#SBATCH --error=AlbanyTesting.cpu.sfad12.%j.err
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --ntasks-per-socket=64
#SBATCH --hint=nomultithread

BASE_DIR=/pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-serial
DEPLOY_DIR=/global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/serial

source ${BASE_DIR}/pm_cpu_gnu_modules.sh >& ${BASE_DIR}/modules_albany.out

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${DEPLOY_DIR}/builds/TrilinosInstall/lib64"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${DEPLOY_DIR}/builds/AlbanyInstallSfad12/lib64"

bash ${BASE_DIR}/nightly_cron_script_mali_pm_cpu.sh

mkdir ${BASE_DIR}/repos/compass_tests
cp ${BASE_DIR}/CTestTestfile.cmake ${BASE_DIR}/repos/compass_tests
cp ${BASE_DIR}/test_and_log.sh ${BASE_DIR}/repos/compass_tests
bash ${BASE_DIR}/nightly_cron_script_compass_pm_cpu.sh

cp biweekly_log_pm_cpu_Compass.txt ${DEPLOY_DIR}/logs/biweekly_log_pm_cpu_Compass.txt
cp biweekly_log_pm_cpu_MALI.txt ${DEPLOY_DIR}/logs/biweekly_log_pm_cpu_MALI.txt

chmod -R 2770 ${DEPLOY_DIR}
chown -R :fanssie ${DEPLOY_DIR}