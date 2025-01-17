#!/bin/csh
BASE_DIR=/pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-cuda-uvm

source ${BASE_DIR}/repos/compass_tests/load_compass_env.sh
compass run $1
result=$?
cat case_outputs/${2}.log
exit $result