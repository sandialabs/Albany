#!/bin/csh

BASE_DIR=/pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-cuda-uvm

cd ${BASE_DIR}/repos/compass_tests

source load_compass_env.sh

compass run

rm -rf /global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/cuda-uvm/compass_tests
cp -R ${BASE_DIR}/repos/compass_tests /global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/cuda-uvm/compass_tests