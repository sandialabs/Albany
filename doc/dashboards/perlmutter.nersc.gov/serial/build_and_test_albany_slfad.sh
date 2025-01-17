#!/bin/bash -login

#SBATCH -A m4274
#SBATCH --job-name=AlbanyTesting
#SBATCH --output=logs/AlbanyTesting.cpu.slfad.%j.out
#SBATCH --error=logs/AlbanyTesting.cpu.slfad.%j.err
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --ntasks-per-socket=64
#SBATCH --hint=nomultithread

BASE_DIR=/pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-serial
DEPLOY_DIR=/global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/serial

source ${BASE_DIR}/pm_cpu_gnu_modules.sh >& ${BASE_DIR}/modules_albany.out
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${DEPLOY_DIR}/builds/TrilinosInstall/lib64"

bash ${BASE_DIR}/nightly_cron_script_albany_pm_cpu.sh slfad none
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${DEPLOY_DIR}/builds/AlbanyInstall/lib64"

cp biweekly_log_pm_cpu_Albany_slfad.txt ${DEPLOY_DIR}/logs/biweekly_log_pm_cpu_Albany_slfad.txt

chmod -R 2770 ${DEPLOY_DIR}
chown -R :fanssie ${DEPLOY_DIR}