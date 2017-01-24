#!/bin/csh

BASE_DIR=/home/ikalash/nightlyCDash
cd $BASE_DIR

export https_proxy="https://wwwproxy.sandia.gov:80"
export http_proxy="http://wwwproxy.sandia.gov:80"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ha_cluster/home/projects/x86-64-sandybridge-nvidia/cuda/7.5.7/lib64:/ha_cluster/home/projects/mpfr/3.1.2/lib:/ha_cluster/home/projects/gmp/5.1.1/lib

cat trilinos ctest_nightly.cmake.frag >& ctest_nightly.cmake  

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=$BASE_DIR/nightly_log_shannonTrilinos.txt

#/opt/local/slurm/default/bin/salloc -N 4 -p stella bash -c \
eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR /home/projects/cmake/2.8.11.2/bin/ctest -VV -S $BASE_DIR/ctest_nightly.cmake" > $LOG_FILE 2>&1

