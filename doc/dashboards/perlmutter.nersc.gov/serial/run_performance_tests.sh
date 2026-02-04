#!/bin/bash -login

#SBATCH -A m4274
#SBATCH --job-name=AlbanyPerfTesting
#SBATCH --output=logs/AlbanyPerfTesting.cpu.%j.out
#SBATCH --error=logs/AlbanyPerfTesting.cpu.%j.err
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=256
#SBATCH --ntasks-per-node=128
#SBATCH --ntasks-per-socket=64
#SBATCH --hint=nomultithread

BASE_DIR=/pscratch/sd/j/jwatkins/nightlyCDashPerlmutterSerial
DEPLOY_DIR=/global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/serial

cd ${BASE_DIR}
source ${BASE_DIR}/pm_cpu_gnu_modules.sh >& ${BASE_DIR}/modules_albany.out

bash nightly_cron_script_ali_perf_tests.sh
bash process_results_ctest.sh 
bash nightly_cron_script_ali_perf_tests_bzip2_save.sh >& biweekly_log_perf_tests_saveresults.txt

cp biweekly_log_perf_tests.txt ${DEPLOY_DIR}/logs/biweekly_log_perf_tests.txt
cp biweekly_log_perf_tests_saveresults.txt ${DEPLOY_DIR}/logs/biweekly_log_perf_tests_saveresults.txt

chmod -R 2770 ${DEPLOY_DIR}
chown -R :fanssie ${DEPLOY_DIR}
