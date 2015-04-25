#!/bin/sh

# Crontab entry
#
# Run at midnight every day (0700 UTC)
#
# 00 07 * * * /home/gahanse/Codes/Albany/doc/dashboards/shannon.sandia.gov/shannon_local/nightly_cron_script.sh

TEST_DIR=/home/gahanse/nightly
SCRIPT_DIR=/home/gahanse/Codes/Albany/doc/dashboards/shannon.sandia.gov/shannon_local

if [ ! -d "$TEST_DIR" ]; then
  mkdir $TEST_DIR
fi

cd $TEST_DIR

export PATH=/home/gahanse/bin:/home/projects/x86-64/openmpi/1.8.4/gnu/4.9.0/cuda/7.0.18/bin:/home/projects/gcc/4.9.0/bin:/home/projects/gmp/5.1.1/bin:/home/projects/mpfr/3.1.2/bin:/home/projects/mpc/1.0.1//bin:/home/projects/x86-64/cuda/7.0.18/bin:/usr/lib64/qt-3.3/bin:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin:/opt/ibutils/bin:/opt/local/slurm/default/bin

export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/home/projects/x86-64/openmpi/1.8.4/gnu/4.9.0/cuda/7.0.18/lib:/home/projects/gcc/4.9.0/lib64:/home/projects/gcc/4.9.0/lib:/home/projects/gmp/5.1.1/lib:/home/projects/mpfr/3.1.2/lib:/home/projects/mpc/1.0.1/lib:/home/projects/x86-64/cuda/7.0.18/lib64:/opt/cray/lib64:/usr/lib64

# Do the proxies to reach the albany github site
export http_proxy=bc-proxy-5.sandia.gov:80
export https_proxy=bc-proxy-5.sandia.gov:80

now=$(date +"%m_%d_%Y-%H_%M")
#LOG_FILE=/projects/AppComp/nightly/cee-compute011/nightly_$now
LOG_FILE=$TEST_DIR/nightly_log_albany.txt

echo "Date and time is $now" > $LOG_FILE

if [ ! -d "$TEST_DIR/buildAlbany" ]; then
  mkdir $TEST_DIR/buildAlbany
fi
if [ ! -d "$TEST_DIR/buildAlbany/nightly" ]; then
  mkdir $TEST_DIR/buildAlbany/nightly
fi
if [ ! -d "$TEST_DIR/buildAlbany/nightly/Albany" ]; then
  mkdir $TEST_DIR/buildAlbany/nightly/Albany
fi

/bin/rm -rf $TEST_DIR/buildAlbany/nightly/Albany/*

salloc -n 4 -N 4 -p stella bash -c \
"env MV2_USE_CUDA=1 TEST_DIRECTORY=$TEST_DIR SCRIPT_DIRECTORY=$SCRIPT_DIR /home/gahanse/bin/ctest -VV -S $SCRIPT_DIR/ctest_nightly.cmake" >> $LOG_FILE 2>&1

/usr/bin/rsync -avz --delete $TEST_DIR/buildAlbany/nightly/Albany/ software-login.sandia.gov:/home/gahanse/Albany > $LOG_FILE 2>&1
