#!/bin/csh

BASE_DIR=/pscratch/sd/j/jwatkins/nightlyCDashPerlmutterSerial

cd ${BASE_DIR}/repos/compass_tests

source load_compass_env.sh

compass run

cp -r ${BASE_DIR}/repos/compass_tests /global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/serial/compass_tests