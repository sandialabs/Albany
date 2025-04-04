#!/bin/bash -login

#SBATCH -A cli193
#SBATCH --job-name=AlbanyPerfTesting
#SBATCH --output=logs/AlbanyPerfTesting.%j.out
#SBATCH --error=logs/AlbanyPerfTesting.%j.err 
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --time=02:00:00 
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest

BASE_DIR=/lustre/orion/cli193/scratch/mcarlson/testingCDashFrontier-rocm
DEPLOY_DIR=/lustre/orion/cli193/proj-shared/automated_testing/rocm

cd $BASE_DIR

source ${BASE_DIR}/frontier_gpu_modules.sh

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

LOG_FILE=$BASE_DIR/test_log_frontier_ali_perf_tests.txt

bash nightly_cron_script_ali_perf_tests.sh
bash process_results_ctest.sh 
bash nightly_cron_script_ali_perf_tests_bzip2_save.sh >& test_log_frontier_ali_perf_tests_saveresults.txt

cp test_log_frontier_ali_perf_tests.txt ${DEPLOY_DIR}/logs/test_log_frontier_ali_perf_tests.txt
cp test_log_frontier_ali_perf_tests_saveresults.txt ${DEPLOY_DIR}/logs/test_log_frontier_ali_perf_tests_saveresults.txt

chmod -R 2770 ${DEPLOY_DIR}
chown -R :fanssie ${DEPLOY_DIR}