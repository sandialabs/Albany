#!/bin/bash -login

#SBATCH -A cli193
#SBATCH --job-name=AlbanyTesting
#SBATCH --output=logs/AlbanyTesting.%j.out
#SBATCH --error=logs/AlbanyTesting.%j.err 
#SBATCH -p batch
#SBATCH -q debug
#SBATCH --nodes=1
#SBATCH --time=02:00:00 
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest

BASE_DIR=/lustre/orion/cli193/scratch/mcarlson/testingCDashFrontier-rocm
cd $BASE_DIR

source ${BASE_DIR}/frontier_gpu_modules.sh

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

LOG_FILE=$BASE_DIR/test_log_frontier_Albany.txt

eval "env BUILD_OR_TEST=test env TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_albany.cmake" > $LOG_FILE 2>&1