#!/bin/sh

# Crontab entry
#
# Run test at 21:00 (9:00PM MDT --> 3:00 UTC, 8:00PM MST --> 3:00 UTC)
# 00 21 * * * /home/gahanse/Codes/Albany/doc/dashboards/shannon.sandia.gov/shannon_local/nightly_cron_script.sh

SUBMIT_RESULTS=ON
#SUBMIT_RESULTS=OFF
THE_TEST_TYPE=Nightly
#THE_TEST_TYPE=Experimental

#/usr/bin/modulecmd bash load openmpi/1.8.4/gnu/4.7.2/cuda/7.0.28
/usr/bin/modulecmd bash load gcc/4.9.0
/usr/bin/modulecmd bash load cuda/7.5.7

export CUDA_LAUNCH_BLOCKING=1

TEST_DIR=/home/gahanse/nightly
SCRIPT_DIR=/home/gahanse/Codes/Albany/doc/dashboards/shannon.sandia.gov/shannon_local

if [ ! -d "$TEST_DIR" ]; then
  /bin/mkdir $TEST_DIR
fi

cd $TEST_DIR

export PATH=/home/gahanse/gcc-4.9.0/mpich-3.1.4/bin:/home/gahanse/bin:/home/projects/gcc/4.9.0/bin:/home/projects/gmp/5.1.1/bin:/home/projects/mpfr/3.1.2/bin:/home/projects/mpc/1.0.1//bin:/home/projects/x86-64-sandybridge-nvidia/cuda/7.5.7/bin:/usr/lib64/qt-3.3/bin:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin:/opt/ibutils/bin:/opt/local/slurm/default/bin
export LD_LIBRARY_PATH=/home/gahanse/gcc-4.9.0/mpich-3.1.4/lib:/home/projects/gcc/4.9.0/lib64:/home/projects/gcc/4.9.0/lib:/home/projects/gmp/5.1.1/lib:/home/projects/mpfr/3.1.2/lib:/home/projects/mpc/1.0.1//lib:/home/projects/x86-64-sandybridge-nvidia/cuda/7.5.7/lib64:/home/gahanse/lib:/opt/cray/lib64:/usr/lib64

#export PATH=/home/gahanse/bin:/home/projects/x86-64/openmpi/1.8.4/gnu/4.7.2/cuda/7.0.28/bin:/home/projects/gcc/4.7.2/bin:/home/projects/gmp/5.1.1/bin:/home/projects/mpfr/3.1.2/bin:/home/projects/mpc/1.0.1//bin:/home/projects/x86-64/cuda/7.0.28/bin:/usr/lib64/qt-3.3/bin:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin:/opt/ibutils/bin:/opt/local/slurm/default/bin
#export LD_LIBRARY_PATH=/home/projects/x86-64/openmpi/1.8.4/gnu/4.7.2/cuda/7.0.28/lib:/home/projects/gcc/4.7.2/lib64:/home/projects/gcc/4.7.2/lib:/home/projects/gmp/5.1.1/lib:/home/projects/mpfr/3.1.2/lib:/home/projects/mpc/1.0.1/lib:/home/projects/x86-64/cuda/7.0.28/lib64:/home/gahanse/lib:/opt/intel/mkl/lib/intel64:/opt/cray/lib64:/usr/lib64

# Do the proxies to reach the albany github site
export http_proxy=bc-proxy-5.sandia.gov:80
export https_proxy=bc-proxy-5.sandia.gov:80

now=$(date +"%m_%d_%Y-%H_%M")
#LOG_FILE=/projects/AppComp/nightly/cee-compute011/nightly_$now
LOG_FILE=$TEST_DIR/nightly_log_albany.txt

echo "Date and time is $now" > $LOG_FILE

if [ ! -d "$TEST_DIR/buildAlbany" ]; then
  /bin/mkdir $TEST_DIR/buildAlbany
fi
if [ ! -d "$TEST_DIR/buildAlbany/nightly" ]; then
  /bin/mkdir $TEST_DIR/buildAlbany/nightly
fi
if [ ! -d "$TEST_DIR/buildAlbany/nightly/Albany" ]; then
  /bin/mkdir $TEST_DIR/buildAlbany/nightly/Albany
else
  /bin/rm -rf $TEST_DIR/buildAlbany/nightly/Albany/*
fi

/opt/local/slurm/default/bin/salloc -n 4 -N 4 -p stella bash -c \
"env MV2_USE_CUDA=1 DO_SUBMIT=$SUBMIT_RESULTS TEST_TYPE=$THE_TEST_TYPE TEST_DIRECTORY=$TEST_DIR SCRIPT_DIRECTORY=$SCRIPT_DIR /home/gahanse/bin/ctest -VV -S $SCRIPT_DIR/ctest_nightly.cmake" >> $LOG_FILE 2>&1

if [ "$SUBMIT_RESULTS" = "ON" ]; then
  /usr/bin/rsync -avz --delete $TEST_DIR/buildAlbany/nightly/Albany/ software-login.sandia.gov:/home/gahanse/Albany >> $LOG_FILE 2>&1
fi
