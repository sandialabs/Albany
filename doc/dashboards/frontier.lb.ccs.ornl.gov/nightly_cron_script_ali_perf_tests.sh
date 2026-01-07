#!/bin/sh

BASE_DIR=/lustre/orion/cli193/scratch/mcarlson/testingCDashFrontier-rocm
DEPLOY_DIR=/lustre/orion/cli193/proj-shared/automated_testing/rocm
cd $BASE_DIR

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

rm -rf *slurm*

export OMP_NUM_THREADS=1

export KOKKOS_TOOLS_LIBS=/lustre/orion/cli193/proj-shared/automated_testing/kokkos-tools/profiling/space-time-stack-mem-only/hip/kp_space_time_stack.so
source ${BASE_DIR}/frontier_gpu_modules.sh >& ${BASE_DIR}/modules_albany.out

printenv |& tee out-env.txt

LOG_FILE=$BASE_DIR/test_log_frontier_ali_perf_tests.txt

eval "env TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR DEPLOY_DIR=$DEPLOY_DIR ctest -VV -S $BASE_DIR/ctest_nightly_perf_tests.cmake" > $LOG_FILE 2>&1