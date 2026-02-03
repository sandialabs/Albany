#!/bin/bash -login

#SBATCH -A m4274_g
#SBATCH --job-name=AlbanyPerfTesting
#SBATCH --output=logs/AlbanyPerfTesting.cuda.%j.out
#SBATCH --error=logs/AlbanyPerfTesting.cuda.%j.err
#SBATCH --constraint=gpu
#SBATCH --nodes=2
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH --gpu-bind=none

BASE_DIR=/pscratch/sd/j/jwatkins/nightlyCDashPerlmutterCuda
DEPLOY_DIR=/global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/cuda

source ${BASE_DIR}/pm_gpu_gnu_modules.sh >& ${BASE_DIR}/modules_albany.out

bash nightly_cron_script_ali_perf_tests.sh
bash process_results_ctest.sh 
bash nightly_cron_script_ali_perf_tests_bzip2_save.sh >& biweekly_log_perf_tests_saveresults.txt

cp biweekly_log_perf_tests.txt ${DEPLOY_DIR}/logs/biweekly_log_perf_tests.txt
cp biweekly_log_perf_tests_saveresults.txt ${DEPLOY_DIR}/logs/biweekly_log_perf_tests_saveresults.txt

chmod -R 2770 ${DEPLOY_DIR}
chown -R :fanssie ${DEPLOY_DIR}