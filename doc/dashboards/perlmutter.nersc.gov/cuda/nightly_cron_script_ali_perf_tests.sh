#!/bin/sh

BASE_DIR=/pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-cuda
DEPLOY_DIR=/global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/cuda
cd $BASE_DIR

unset http_proxy
unset https_proxy

rm -rf *slurm*

export OMP_NUM_THREADS=1

export KOKKOS_TOOLS_LIBS=/global/cfs/cdirs/fanssie/software/kokkos-tools/kp_space_time_stack.so
source ${BASE_DIR}/pm_gpu_gnu_modules.sh >& ${BASE_DIR}/modules_albany.out 

printenv |& tee out-env.txt

LOG_FILE=$BASE_DIR/biweekly_log_perf_tests.txt

eval "env TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_perf_tests.cmake" > $LOG_FILE 2>&1