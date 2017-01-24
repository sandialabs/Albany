#!/bin/csh

BASE_DIR=/home/ikalash/nightlyCDash
cd $BASE_DIR

export https_proxy="https://wwwproxy.sandia.gov:80"
export http_proxy="http://wwwproxy.sandia.gov:80"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ha_cluster/home/projects/x86-64-sandybridge-nvidia/cuda/7.5.7/lib64:/ha_cluster/home/projects/mpfr/3.1.2/lib:/ha_cluster/opt/intel/composer_xe_2013.3.163/mkl/lib/intel64/

cat albany ctest_nightly.cmake.frag >& ctest_nightly.cmake  

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=$BASE_DIR/nightly_log_shannonAlbany.txt

#eval 
/opt/local/slurm/default/bin/salloc -N 4 -p atlas bash -c \
"env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR  ctest -VV -S $BASE_DIR/ctest_nightly.cmake" > $LOG_FILE 2>&1

bash process_results_ctest.sh
bash send_email_ctest.sh

