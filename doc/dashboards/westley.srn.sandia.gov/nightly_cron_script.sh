#!/bin/bash

# Crontab entry
#
# Run at midnight every day (0000 MDT)
#
#00 00 * * * /home/gahanse/Codes/Albany/doc/dashboards/westley.srn.sandia.gov/nightly_cron_script.sh


#set -o xtrace

SUBMIT_RESULTS=ON
#SUBMIT_RESULTS=OFF
THE_TEST_TYPE=Nightly
#THE_TEST_TYPE=Experimental

TEST_DIR=/home/gahanse/nightly
SCRIPT_DIR=/home/gahanse/Codes/Albany/doc/dashboards/westley.srn.sandia.gov

if [ ! -d "$TEST_DIR" ]; then
  /bin/mkdir $TEST_DIR
fi

cd $TEST_DIR

. /opt/intel/bin/compilervars.sh intel64
export PATH=/usr/local/trilinos/MPI_REL/bin:/opt/intel/bin:/usr/local/bin:/usr/lib64/qt-3.3/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin:/home/gahanse/bin
export LD_LIBRARY_PATH=/opt/intel/compilers_and_libraries/linux/lib/mic:/usr/local/gcc-5.1.0/lib64:$LD_LIBRARY_PATH

export I_MPI_MIC=1

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

eval "env DO_SUBMIT=$SUBMIT_RESULTS TEST_TYPE=$THE_TEST_TYPE TEST_DIRECTORY=$TEST_DIR SCRIPT_DIRECTORY=$SCRIPT_DIR /usr/local/bin/ctest -VV -S $SCRIPT_DIR/ctest_nightly.cmake" >> $LOG_FILE 2>&1
