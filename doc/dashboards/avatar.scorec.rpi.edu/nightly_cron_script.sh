#!/bin/sh

now=$(date +"%m_%d_%Y")
LOG_FILE=/fasttmp/ghansen/nightly/nightly_$now

if [ ! -d /fasttmp/ghansen/nightly/repos ]; then
  mkdir /fasttmp/ghansen/nightly/repos
fi

if [ ! -d /fasttmp/ghansen/nightly/build ]; then
  mkdir /fasttmp/ghansen/nightly/build
fi


env  TEST_DIRECTORY=/fasttmp/ghansen/nightly \
     ctest -VV -S /fasttmp/ghansen/Albany/doc/dashboards/avatar.scorec.rpi.edu/ctest_nightly_mpi_opt_avatar.cmake
     > $LOG_FILE 2>&1

