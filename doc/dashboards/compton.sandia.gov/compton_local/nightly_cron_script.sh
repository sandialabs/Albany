#!/bin/bash

# Crontab entry
#
# Run at midnight every day (0000 MDT)
#
# 00 00 * * * /home/gahanse/Codes/Albany/doc/dashboards/compton.sandia.gov/compton_local/nightly_cron_script.sh

SUBMIT_RESULTS=ON
#SUBMIT_RESULTS=OFF
THE_TEST_TYPE=Nightly
#THE_TEST_TYPE=Experimental

TEST_DIR=/home/gahanse/nightly
SCRIPT_DIR=/home/gahanse/Codes/Albany/doc/dashboards/compton.sandia.gov/compton_local

if [ ! -d "$TEST_DIR" ]; then
  /bin/mkdir $TEST_DIR
fi

cd $TEST_DIR

export PATH=/home/software/intel/ics_install/impi/4.1.1.036/mic/bin:/home/projects/x86-64/intel/compilers/2015/composer_xe_2015.2.164/bin:/home/projects/x86-64/intel/compilers/2015/composer_xe_2015.2.164/mpirt/bin/intel64:/home/projects/x86-64/intel/compilers/2015/composer_xe_2015.2.164/bin/intel64:/home/projects/x86-64/intel/compilers/2015/composer_xe_2015.2.164/debugger/gui/intel64:/home/projects/x86-64/intel/compilers/2015/composer_xe_2015.2.164/bin/intel64_mic:/home/projects/gcc/4.7.2/bin:/home/projects/gmp/5.0.5/bin:/home/projects/mpfr/3.1.0/bin:/home/projects/mpc/1.0.1/bin:/home/projects/x86-64/cmake/3.0.2/bin:/usr/lib64/qt-3.3/bin:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin:/home/gahanse/bin

export LD_LIBRARY_PATH=/home/projects/x86-64-knc/boost/1.58.0/intel/15.2.164/intelmpi/4.1.1.036/lib:/home/software/intel/ics_install/impi/4.1.1.036/mic/lib:/home/projects/x86-64/intel/compilers/2015/composer_xe_2015.2.164/tbb/lib/intel64:/home/projects/x86-64/intel/compilers/2015/composer_xe_2015.2.164/mkl/lib/intel64:/opt/intel/mic/coi/host-linux-release/lib:/opt/intel/mic/myo/lib:/home/projects/x86-64/intel/compilers/2015/composer_xe_2015.2.164/ipp/lib/intel64:/home/projects/x86-64/intel/compilers/2015/composer_xe_2015.2.164/mpirt/lib/intel64:/home/projects/x86-64/intel/compilers/2015/composer_xe_2015.2.164/compiler/lib/intel64:/home/projects/x86-64/intel/compilers/2015/composer_xe_2015.2.164/lib64:/home/projects/x86-64/intel/compilers/2015/composer_xe_2015.2.164/lib:/home/projects/gcc/4.7.2/lib64:/home/projects/gcc/4.7.2/lib:/home/projects/gmp/5.0.5/lib:/home/projects/mpfr/3.1.0/lib:/home/projects/mpc/1.0.1//lib:/home/projects/x86-64/cmake/3.0.2/lib

export MIC_LD_LIBRARY_PATH=/home/projects/x86-64/intel/compilers/2015/composer_xe_2015.2.164/tbb/lib/mic:/home/projects/x86-64/intel/compilers/2015/composer_xe_2015.2.164/mkl/lib/mic:/opt/intel/mic/myo/lib:/opt/intel/mic/coi/device-linux-release/lib:/home/projects/x86-64/intel/compilers/2015/composer_xe_2015.2.164/compiler/lib/mic

export LIBRARY_PATH=/home/projects/x86-64-knc/boost/1.58.0/intel/15.2.164/intelmpi/4.1.1.036/lib
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export I_MPI_FABRICS=shm:dapl
export IPPROOT=/home/projects/x86-64/intel/compilers/2015/composer_xe_2015.2.164/ipp
export MKLROOT=/home/projects/x86-64/intel/compilers/2015/composer_xe_2015.2.164/mkl
export BOOST_ROOT=/home/projects/x86-64-knc/boost/1.58.0/intel/15.2.164/intelmpi/4.1.1.036
export I_MPI_ROOT=/home/software/intel/ics_install/impi/4.1.1.036

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


#/opt/local/slurm/default/bin/salloc -n 4 -N 4 -p stella bash -c \
#"env MV2_USE_CUDA=1 DO_SUBMIT=$SUBMIT_RESULTS TEST_TYPE=$THE_TEST_TYPE TEST_DIRECTORY=$TEST_DIR SCRIPT_DIRECTORY=$SCRIPT_DIR /home/gahanse/bin/ctest -VV -S $SCRIPT_DIR/ctest_nightly.cmake" >> $LOG_FILE 2>&1

#/usr/local/bin/salloc -n 4 -N 4 bash -c \
bash -c \
"env DO_SUBMIT=$SUBMIT_RESULTS TEST_TYPE=$THE_TEST_TYPE TEST_DIRECTORY=$TEST_DIR SCRIPT_DIRECTORY=$SCRIPT_DIR /home/projects/x86-64/cmake/3.0.2/bin/ctest -VV -S $SCRIPT_DIR/ctest_nightly.cmake" >> $LOG_FILE 2>&1

if [ "$SUBMIT_RESULTS" = "ON" ]; then
  /usr/bin/rsync -avz --delete $TEST_DIR/buildAlbany/nightly/Albany/ software-login.sandia.gov:/home/gahanse/Albany_compton >> $LOG_FILE 2>&1
fi
