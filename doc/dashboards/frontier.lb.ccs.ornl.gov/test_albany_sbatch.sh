#!/bin/bash -login

#SBATCH -A cli193
#SBATCH --job-name=AlbanyTesting
#SBATCH --output=logs/AlbanyTesting.%j.out
#SBATCH --error=logs/AlbanyTesting.%j.err 
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

FAD_CONFIGURATION="slfad"
FAD_SIZE="none"

LOG_FILE=$BASE_DIR/test_log_frontier_Albany.txt

eval "BUILD_OR_TEST=test FAD_CONFIGURATION=${FAD_CONFIGURATION} FAD_SIZE=${FAD_SIZE} BASE_DIR=${BASE_DIR} DEPLOY_DIR=${DEPLOY_DIR} TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_albany_slfad.cmake" > $LOG_FILE 2>&1

cp test_log_frontier_Albany.txt ${DEPLOY_DIR}/logs/test_log_frontier_Albany.txt
chmod -R 2770 ${DEPLOY_DIR}
chown -R :cli193 ${DEPLOY_DIR}

